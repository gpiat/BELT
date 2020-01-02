import constants as cst
import math
import pickle
import torch

from sys import argv
from util import get_start_end_indices
from util import get_text_window


def evaluate(model, corpus, umls_concepts, numericalizer,
             window_size=20, batch_size=35):
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for document in corpus.documents():
            text = numericalizer.numericalize_text(document.text)
            for i in range(len(text)):
                s_idx, e_idx = get_start_end_indices(i, len(text), window_size)
                data = get_text_window(text, window_size, s_idx, e_idx)

                target = torch.Tensor(
                    [umls_concepts[document.get_cuid(i)]]
                ).long().to(cst.device)
                # targets = torch.zeros(len(umls_concepts))
                # targets[target] = 1

                output = model(data.unsqueeze(0), target_words=torch.Tensor(
                    [text[i]]).to(cst.device))
                print("output shape in eval: ", output.shape)
                print("target shape in eval: ", target.shape)
                total_loss += (len(data) *
                               cst.criterion(output, target).item())

    return total_loss / (corpus.n_documents - 1)


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
    test_loss = evaluate(best_model, test_corpus,
                         numericalizer, umls_concepts)
    print('=' * 89)
    try:
        test_ppl = math.exp(test_loss)
    except OverflowError:
        print("Test ppl too large to compute")
        test_ppl = 0
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, test_ppl))
    print('=' * 89)
