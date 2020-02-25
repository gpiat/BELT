import constants as cst
import math
import pickle
import torch

from sys import argv
from util import get_start_end_indices
from util import get_text_window


def cuid_list_to_ranges(cuids):
    """ Args:
            cuids (list<int>): list of CUIDs, likely with repetitions
        Returns:
            ranges (list<[int, int, int]>): list of ranges such that each
                range is a list of the form <begin, end, CUID>.
        Example:
            In:     0 0 0 0 4 4 4 4 4 0 1 1 1
                    ^     ^ ^       ^ ^ ^   ^
                    |     |beg     end| |   |
                   beg   end          |beg end
                                   beg+end
            Out: [(0, 3, 0), (4, 8, 4), (9, 9, 0), (10, 12, 1)]
    """
    ranges = [[0, 0, cuids[0]]]
    for i in range(1, len(cuids)):
        if cuids[i] == cuids[i - 1]:
            ranges[-1][1] = i
        else:
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
        predictions_i_ranges = cuid_list_to_ranges(predictions[i])
        targets_i_ranges = cuid_list_to_ranges(targets[i])

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
    #         err_msg = ("No condition could determine whether the case is of" +
    #                    " a True Positive / True Negative / False Positive /" +
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


def evaluate(model, corpus, umls_concepts, numericalizer,
             window_size=20, compute_p_r_f1=False):
    """ Evaluates a BELT model
        Args:
            - (TransformerModel) model: the model to evaluate
            - (MedMentionsCorpus) corpus: the evaluation corpus
            - (dict) umls_concepts: a dict mapping UMLS CUIDs to
                indices. Assumes all the CUIDs used in the corpus
                are indexed in umls_concepts.
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
            document_tagged = []
            document_targets = []
            text = numericalizer.numericalize_text(document.text)
            for i in range(len(text)):
                s_idx, e_idx = get_start_end_indices(i, len(text), window_size)
                data = get_text_window(text, window_size, s_idx, e_idx)

                target = torch.Tensor(
                    [umls_concepts[document.get_cuid(i)]]
                ).long().to(cst.device)
                document_targets.append(int(target))

                output = model(data.unsqueeze(0), target_words=torch.Tensor(
                    [text[i]]).to(cst.device))
                document_tagged.append(int(torch.argmax(output)))

                total_loss += (len(data) *
                               cst.criterion(output, target).item())
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
    try:
        _, test_fname, model_fname = argv
    except Exception:
        test_fname = cst.test_fname
        model_fname = cst.model_fname

    with open(cst.umls_fname, 'rb') as umls_con_file:
        umls_concepts = pickle.load(umls_con_file)
    with open(cst.numer_fname, 'rb') as numericalizer_file:
        numericalizer = pickle.load(numericalizer_file)
    with open(test_fname, 'rb') as test_file:
        test_corpus = pickle.load(test_file)
    with open(model_fname, 'rb') as model_file:
        best_model = pickle.load(model_file)

    # start test
    (test_loss,
     (mention_precision, mention_recall, mention_f1),
     (doc_precision, doc_recall, doc_f1)) =\
        evaluate(best_model, test_corpus, umls_concepts,
                 numericalizer, compute_p_r_f1=True)

    print('=' * 89)
    try:
        test_ppl = math.exp(test_loss)
    except OverflowError:
        print("Test ppl too large to compute")
        test_ppl = 0
    print('End of training')
    print('test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_ppl))
    print('mention precision {:5.2f} | mention recall {:5.2f} |\
 mention f1 {:8.2f}'.format(mention_precision, mention_recall, mention_f1))
    print('document precision {:5.2f} | document recall {:5.2f} |\
 document f1 {:8.2f}'.format(doc_precision, doc_recall, doc_f1))
    print('=' * 89)
