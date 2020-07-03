import constants as cst
import pickle
import torch

from sys import argv
from util import get_text_window
from util import pad
from util import parse_args


def cuid_list_to_ranges(cuids):
    """ Args:
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
    tp = 0
    # tn = 0
    fp = 0
    fn = 0

    for i in range(len(predictions)):  # == len(targets)
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

    # for prediction, target in zip(chain(*predictions), chain(*targets)):
    #     if target == prediction == 0:
    #         tn += 1
    #     elif target == prediction != 0:
    #         tp += 1
    #     elif target == 0 != prediction:
    #         fp += 1
    #     elif target != 0 == prediction:
    #         fn += 1
    #     elif 0 != target != prediction != 0:
    #         fp += 1
    #         fn += 1
    #     else:
    #         err_msg = ("No condition could determine whether the case is of"
    #                    " a True Positive / True Negative / False Positive /"
    #                    " False Negative. Prediction: " + str(prediction) +
    #                    " Target: " + str(target))
    #         raise ValueError(err_msg)

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


def predict(model, document, umls_cuid_to_idx,
            numericalizer, overlap):
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
        # s_idx, e_idx = get_start_end_indices(i, len(text), window_size)
        data = get_text_window(text, window_size, s_idx, e_idx)

        target = torch.Tensor(
            [umls_cuid_to_idx[document.get_cuid(i)]]
        ).long().to(cst.device)
        document_targets.append(int(target))

        output = model(data.unsqueeze(0))
        # , target_words=torch.Tensor([text[i]]).to(cst.device))

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
        stop = len(output) - start
        # Special cases: the very beginning and end of the text do not
        # overlap with another window, therefore they cannot be discarded.
        if i == 0:
            start = 0
        if i + window_size == len(text):
            stop = len(output)
        # shape of output[j]: [C, 1], basically a vector.
        # the explicit 2nd dimension of the Tensor isn't a problem
        # when dealing with argmax.
        document_tagged += [int(torch.argmax(output[j]))
                            for j in range(start, stop)]

        loss_increment = (len(data) * cst.criterion(output, target).item())
    return document_tagged, document_targets, loss_increment


def evaluate(model, corpus, umls_cuid_to_idx, numericalizer,
             txt_window_overlap, compute_p_r_f1=False):
    """ Evaluates a BELT model
        Args:
            - (TransformerModel) model: the model to evaluate
            - (MedMentionsCorpus) corpus: the evaluation corpus
            - (dict) umls_cuid_to_idx: a dict mapping UMLS CUIDs to
                indices. Assumes all the CUIDs used in the corpus
                are indexed in umls_cuid_to_idx.
            - (util.Numericalizer) numericalizer: converts words to
                numbers for the purpose of input to the model. This
                should be consistent with the numericalizer that was
                used in training.
            - (int) window_size: number of words that should be considered
                at a time (defaults to 20). Should be consistent with
                the length of phrases at training time.
    """
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    text_tagged = []
    text_targets = []
    with torch.no_grad():
        for document in corpus.documents():
            document_tagged, document_targets, loss_increment =\
                predict(model, document, umls_cuid_to_idx,
                        numericalizer, txt_window_overlap)
            total_loss += loss_increment
            text_tagged.append(document_tagged)
            text_targets.append(document_targets)

    loss = total_loss / (corpus.n_documents - 1)
    if compute_p_r_f1:
        return (loss,
                get_mention_prec_rec_f1(text_tagged, text_targets),
                get_document_prec_rec_f1(text_tagged, text_targets))
    else:
        return loss


if __name__ == '__main__':
    args = {}
    args['--test_fname'] = cst.test_fname
    args['--model_fname'] = cst.model_fname
    args['--umls_fname'] = cst.umls_fname
    args['--numer_fname'] = cst.numer_fname
    args['--predictions_fname'] = cst.wd + "predictions.out"
    args['--targets_fname'] = cst.wd + "targets.out"
    args['--write_pred'] = False
    args['--skip_eval'] = False
    args['--overlap'] = 0.2
    # args['--window_size'] = 20

    parse_args(argv, args)
    args['--overlap'] = float(args['--overlap'])
    # args['--window_size'] = int(args['--window_size'])

    with open(args['--umls_fname'], 'rb') as umls_con_file:
        umls_cuid_to_idx = pickle.load(umls_con_file)
    with open(args['--numer_fname'], 'rb') as numericalizer_file:
        numericalizer = pickle.load(numericalizer_file)
    with open(args['--test_fname'], 'rb') as test_file:
        test_corpus = pickle.load(test_file)
    with open(args['--model_fname'], 'rb') as model_file:
        best_model = pickle.load(model_file)

    if args['--write_pred']:
        umls_idx_to_cuid = {v: k for k, v in umls_cuid_to_idx.items()}
        best_model.eval()  # Turn on the evaluation mode
        with torch.no_grad():
            print("number of documents: ", test_corpus.n_documents)
            for document in test_corpus.documents():
                document_tagged, document_targets, _ =\
                    predict(best_model, document, umls_cuid_to_idx,
                            numericalizer, args['--overlap'])
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
         (doc_precision, doc_recall, doc_f1)) =\
            evaluate(best_model, test_corpus,
                     umls_cuid_to_idx, numericalizer,
                     args['--overlap'], compute_p_r_f1=True)

        print('=' * 89)
        print('test loss {:5.2f}'.format(test_loss))
        print('mention precision {:5.2f} | mention recall {:5.2f} |\
     mention f1 {:8.2f}'.format(mention_precision, mention_recall, mention_f1))
        print('document precision {:5.2f} | document recall {:5.2f} |\
     document f1 {:8.2f}'.format(doc_precision, doc_recall, doc_f1))
        print('=' * 89)
