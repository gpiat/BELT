import constants as cst
import math
import pickle
import torch

from itertools import chain
from sys import argv
from util import get_start_end_indices
from util import get_text_window


def get_prec_rec_f1(predictions, targets):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for prediction, target in zip(chain(*predictions), chain(*targets)):
        if target == prediction == 0:
            tn += 1
        elif target == prediction != 0:
            tp += 1
        elif target == 0 != prediction:
            fp += 1
        elif target != 0 == prediction:
            fn += 1
        elif 0 != target != prediction != 0:
            fp += 1
            fn += 1
        else:
            err_msg = ("No condition could determine whether the case is of" +
                       " a True Positive / True Negative / False Positive /" +
                       " False Negative. Prediction: " + str(prediction) +
                       " Target: " + str(target))
            raise ValueError(err_msg)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
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
                document_targets.append(target)

                output = model(data.unsqueeze(0), target_words=torch.Tensor(
                    [text[i]]).to(cst.device))
                document_tagged.append(int(torch.argmax(output)))

                total_loss += (len(data) *
                               cst.criterion(output, target).item())
            text_tagged.append(document_tagged)
            text_targets.append(document_targets)

    loss = total_loss / (corpus.n_documents - 1)
    if compute_p_r_f1:
        return loss, get_prec_rec_f1(text_tagged, text_targets)
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
    test_loss, (precision, recall, f1) =\
        evaluate(best_model, test_corpus, umls_concepts,
                 numericalizer, compute_p_r_f1=True)

    print('=' * 89)
    try:
        test_ppl = math.exp(test_loss)
    except OverflowError:
        print("Test ppl too large to compute")
        test_ppl = 0
    print('| End of training | test loss {:5.2f} |\
 test ppl {:8.2f} | precision {:5.2f} | recall {:5.2f} |\
 f1 {:8.2f}'.format(test_loss, test_ppl, precision, recall, f1))
    print('=' * 89)
