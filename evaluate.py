import numpy as np
import pickle
import torch

from sys import argv, exit

import constants as cst
from args_handler import get_evaluate_args
from util import get_text_window
from util import pad
from util import set_targets


def cuid_list_to_ranges(cuids):
    """ TODO: RENAME THIS FUNCTION
              it no longer handles only CUIDs,
              it also handles other class labels
        Args:
            cuids (list<int>): list of CUIDs, likely with repetitions
        Returns:
            ranges (list<[int, int, int]>): list of ranges such that each
                range is a list of the form [begin, end + 1, CUID].
        Example:
            In:     0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 1, 1, 1
                    ^        ^  ^           ^  ^  ^     ^
                    |        | beg         end |  |     |
                   beg      end                | beg   end
                                            beg+end
            Out: [[0, 4, 0], [4, 9, 4], [9, 10, 0], [10, 12, 1]]
    """
    ranges = [[0, 0, cuids[0]]]
    for i in range(1, len(cuids)):
        ranges[-1][1] = i
        if cuids[i] != cuids[i - 1]:
            ranges.append([i, i, cuids[i]])
    return ranges


def prec_rec_f1(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return precision, recall, f1


def get_mention_prec_rec_f1(predictions, targets):
    """ Computes precision, recall and F1 at mention level.
        The original paper computes true positives, false positives and false
        negatives based on ranges of characters, CUIDs and PMIDs. Here, the
        PMID is not needed because the structure of the `predictions` and
        `targets` lists already accounts for document differentiation, and
        character-identifying indexes are replaced with word indexes.
        These are in theory entirely equivalent.
        Args:
            predictions (list<list<int>>): each int represents the PMID of
                a word, each list<int> represents a document.
            targets (list<list<int>>): each int represents the PMID of
                a word, each list<int> represents a document.
        Return:
            precision (float)
            recall (float)
            f1 (float)
    """
    tp = 0  # True Positives
    # tn = 0, not needed
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for i in range(len(predictions)):  # is = to len(targets)
        predictions_i_ranges = [j for j in cuid_list_to_ranges(
            predictions[i]) if j[2] != 0]
        targets_i_ranges = [j for j in cuid_list_to_ranges(
            targets[i]) if j[2] != 0]

        for prediction in predictions_i_ranges:
            if prediction in targets_i_ranges:
                tp += 1
            else:
                fp += 1
        for target in targets_i_ranges:
            if target not in predictions_i_ranges:
                fn += 1

    return prec_rec_f1(tp, fp, fn)


def get_document_prec_rec_f1(predictions, targets):
    """ Computes precision, recall and F1 at document level.
        Args:
            predictions (list<list<int>>): each int represents the PMID of
                a word, each list<int> represents a document.
            targets (list<list<int>>): each int represents the PMID of
                a word, each list<int> represents a document.
        Return:
            precision (float)
            recall (float)
            f1 (float)
    """
    tp = 0
    fp = 0
    fn = 0
    for doc_predictions, doc_targets in zip(predictions, targets):
        for prediction in set(doc_predictions):
            if prediction in doc_targets:
                tp += 1
            else:
                fp += 1
        for target in set(doc_targets):
            if target not in predictions:
                fn += 1
    return prec_rec_f1(tp, fp, fn)


def get_token_prec_rec_f1(predictions, targets):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    print("getting token stats")
    print("predictions:\n", predictions)
    print("targets:\n", targets)
    predictions = torch.Tensor(predictions)
    targets = torch.Tensor(targets)
    print("predictions as tensor:\n", predictions)
    print("targets as tensor:\n", targets)
    tp = np.logical_and(predictions, targets).sum()
    tn = np.logical_and(np.logical_not(predictions),
                        np.logical_not(targets)).sum()
    fp = np.logical_and(predictions, np.logical_not(targets)).sum()
    fn = np.logical_and(np.logical_not(predictions), targets).sum()
    # pcpt stands for proportion_correctly_predicted_tokens
    pcpt = (tp + tn) / (tp + tn + fp + fn)
    return (*prec_rec_f1(tp, fp, fn), pcpt)


def predict(model, document, target_finder,
            label_to_idx, numericalizer, overlap):
    """ Get predictions for a given model
        Args:
            model: The model doing the prediction
            document <MedMentionsDocument>: The document to annotate
                (a label is predicted for each token in the document)
            target_finder <func>: function which finds the targets for
                a given length of text.
            label_to_idx <dict>: a lookup table associating the true
                label of a token (str) to the index of its corresponding
                index in predicted probability distribution.
            numericalizer: object that manages the translation from text
                to numbers for the model to use
            overlap [0-1]: amount of overlap between two text windows
                preceding and following any given text window.
    """
    window_size = model.phrase_len
    document_tagged = []
    document_targets = []
    text = numericalizer.numericalize_text(pad(document.text,
                                               window_size,
                                               overlap))

    increment = round((1 - (overlap / 2)) * window_size)
    for i in range(0, len(text), increment):
        s_idx = max(min(i, len(text) - window_size), 0)
        e_idx = min(len(text), window_size + i)
        data = get_text_window(text, window_size, s_idx, e_idx)

        target = target_finder(document, i, i + window_size, label_to_idx)

        output = model(data.unsqueeze(0))

        # output shape: [minibatch=1, C, window_size]
        # with C the number of classes for the classification problem
        output = output.permute(2, 1, 0)
        # output shape: [window_size, C, 1]

        # The whole point of the overlapping windows is that with the
        # fixed window, we don't have wnough bidirectional context at
        # the beginning and end. This is where we figure out which
        # predictions to keep and which to trash

        # General case: Each window looks like this:
        # with O meaning "overlapping token" and N meaning "Normal token"
        # O O O O N N N N N N N N N N N N N N N N O O O O
        # Here, there are eight Os and 16 Ns. The overlap is 1/3
        # Each block of Os accounts for half of the overlap
        # We keep the inner halves of each block of overlap and discard
        # the outer halves as such:     (D for Discard, K for Keep)
        # D D K K K K K K K K K K K K K K K K K K K K D D
        start = round((overlap / 4) * window_size)
        stop = window_size - start
        # Special cases: the very beginning and end of the text do not
        # overlap with another window, therefore they cannot be discarded.
        if i == 0:
            start = 0
        if i + window_size == len(text):
            stop = window_size
        # shape of output[j]: [C, 1], basically a vector.
        # the explicit 2nd dimension of the Tensor isn't a problem
        # when dealing with argmax.
        print("output shape:")
        print(output.shape)
        print("start: {}, stop: {}".format(start, stop))
        document_tagged += [int(torch.argmax(output[j]))
                            for j in range(start, stop)]
        print("document tagged length: {}".format(len(document_tagged)))
        document_targets.extend([target[j] for j in range(start, stop)])
        print("document targets length: {}".format(len(document_targets)))
        target = torch.Tensor(target).to(cst.device)

        loss_increment = (
            len(data) * cst.criterion(output,
                                      target.unsqueeze(1).long()).item())
    return document_tagged, document_targets, loss_increment


def evaluate(model, corpus, target_finder, label_to_idx, numericalizer,
             txt_window_overlap, compute_mntn_p_r_f1=False,
             compute_doc_p_r_f1=False, compute_tkn_p_r_f1=False):
    """ Evaluates a BELT model
        Args:
            - (TransformerModel) model: the model to evaluate
            - (MedMentionsCorpus) corpus: the evaluation corpus
            - (func) target_finder: function that finds the prediction
                targets for the given document and text window.
            - (dict) label_to_idx: a dict mapping token labels (UMLS
                CUIDs, STIDs etc.) to indices. Assumes all the CUIDs
                used in the corpus are indexed in label_to_idx.
            - (util.Numericalizer) numericalizer: converts words to
                numbers for the purpose of input to the model. This
                should be consistent with the numericalizer that was
                used in training.
            - (float) txt_window_overlap[0-1]: amount of overlap between two
                text windows preceding and following any given text window.
            - (bool) compute_mntn_p_r_f1: if True, add precision, recall
                and F1 to return value (computed at "mention" level, i.e.
                True Positive => PMID, start index, end index and category
                of mention detected accurately)
            - (bool) compute_doc_p_r_f1: if True, add precision, recall
                and F1 to return value (computed at "document" level, i.e.
                True Positive => PMID, and category of mention detected
                accurately)
            - (bool) compute_tkn_p_r_f1: if True, add precision, recall
                F1 and accuracy to return value (computed at "token" level,
                i.e. True Positive => PMID, single token index, and category
                of mention detected accurately)

    """
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    text_tagged = []
    text_targets = []
    with torch.no_grad():
        for document in corpus.documents():
            print("evaluating document {}".format(document.pmid))
            document_tagged, document_targets, loss_increment =\
                predict(model, document, target_finder,
                        label_to_idx, numericalizer,
                        txt_window_overlap)
            total_loss += loss_increment
            text_tagged.append(document_tagged)
            text_targets.append(document_targets)

    print("text tagged:\n", text_tagged)
    print("text targets:\n", text_targets)

    loss = total_loss / (corpus.n_documents - 1)
    return_val = (loss,)
    if compute_mntn_p_r_f1:
        return_val += (get_mention_prec_rec_f1(text_tagged, text_targets),)
    if compute_doc_p_r_f1:
        return_val += (get_document_prec_rec_f1(text_tagged, text_targets),)
    if compute_tkn_p_r_f1:
        return_val += (get_token_prec_rec_f1(text_tagged, text_targets),)
    return return_val


if __name__ == '__main__':
    args = get_evaluate_args(argv)

    with open(args['--umls_fname'], 'rb') as umls_con_file:
        # TODO: GENERALIZE TO STIDs & BIN
        umls_cuid_to_idx = pickle.load(umls_con_file)

    with open(args['--numer_fname'], 'rb') as numericalizer_file:
        numericalizer = pickle.load(numericalizer_file)
    with open(args['--test_fname'], 'rb') as test_file:
        test_corpus = pickle.load(test_file)
    with open(args['--model_fname'], 'rb') as model_file:
        best_model = pickle.load(model_file)

    target_finder = set_targets(args['--target_type'])

    if args['--write_pred']:
        umls_idx_to_cuid = {v: k for k, v in umls_cuid_to_idx.items()}
        best_model.eval()  # Turn on the evaluation mode
        with torch.no_grad():
            print("number of documents: ", test_corpus.n_documents)
            for document in test_corpus.documents():
                document_tagged, document_targets, _ =\
                    predict(best_model, document, target_finder,
                            umls_cuid_to_idx, numericalizer,
                            args['--overlap'])
                document_tagged = cuid_list_to_ranges(document_tagged)
                document_targets = cuid_list_to_ranges(document_targets)
                for i in document_tagged:
                    # inserting the PMID at the beginning of each range
                    i.insert(0, document.pmid)
                    # and replacing the index of the UMLS concept with its CUID
                    i[-1] = umls_idx_to_cuid[i[-1]]
                for i in document_targets:
                    i.insert(0, document.pmid)
                    i[-1] = umls_idx_to_cuid[i[-1]]
                with open(args['--predictions_fname'], 'a') as f:
                    for tag in document_tagged:
                        print(tag, file=f)
                    print('', file=f)
                with open(args['--targets_fname'], 'a') as f:
                    for target in document_targets:
                        print(target, file=f)
                    print('', file=f)

    if not args['--skip_eval']:

        # start test
        (test_loss,
         (mention_precision, mention_recall, mention_f1),
         (doc_precision, doc_recall, doc_f1),
         (tok_precision, tok_recall, tok_f1, tok_accuracy)) =\
            evaluate(best_model, test_corpus, target_finder,
                     umls_cuid_to_idx, numericalizer,
                     args['--overlap'], compute_mntn_p_r_f1=True,
                     compute_doc_p_r_f1=True, compute_tkn_p_r_f1=True)

        print('=' * 89)
        print('test loss {:5.2f}'.format(test_loss))
        print('mention precision {:5.2f} | mention recall {:5.2f} |'
              'mention f1 {:8.2f}'.format(mention_precision,
                                          mention_recall,
                                          mention_f1))
        print('document precision {:5.2f} | document recall {:5.2f} |'
              'document f1 {:8.2f}'.format(doc_precision,
                                           doc_recall,
                                           doc_f1))
        print('=' * 89)
        print('token precision {:5.2f} | token recall {:5.2f} |'
              'token f1 {:8.2f} | token accuracy {:5.2f}'.format(tok_precision,
                                                                 tok_recall,
                                                                 tok_f1,
                                                                 tok_accuracy))
