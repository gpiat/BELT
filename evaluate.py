import constants as cst
import numpy as np
import os
import pickle
import sys
import torch

from args_handler import get_evaluate_args
from dataset import extract_label_mapping
from dataset import collate_ner

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from statistics import mean
from sys import argv
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


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
    """ Gets precision, recall, F1, and proportion of correctly predicted
        tokens at the token level.
        Args:
            predictions (list<list>): should contain one list per document.
                Each sublist should contain token-level predictions.
            targets (list<list>): should contain one list per document.
                Each sublist should contain token-level targets.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    # len(predictions) = len(targets) = number of documents
    # we must handle each sublist individually since they
    # are of size <number of tokens in document>, meaning
    # they have very different sizes.
    for i in range(len(predictions)):
        pred_i = torch.Tensor(predictions[i])
        target_i = torch.Tensor(targets[i])
        # a true positive is when the prediction matches the target AND
        # a positive was predicted (i.e. not  0)
        tp += np.logical_and(np.equal(pred_i, target_i),
                             np.not_equal(pred_i,
                                          np.zeros_like(pred_i))).sum()
        # a true negative is when both the target and the prediction
        # are negatives
        tn += np.logical_and(np.logical_not(pred_i),
                             np.logical_not(target_i)).sum()
        # a false positive is when the prediction does not match the
        # target AND a positive was predicted (i.e. not  0)
        fp += np.logical_and(np.not_equal(pred_i, target_i),
                             np.not_equal(pred_i,
                                          np.zeros_like(pred_i))).sum()
        # a false negative is when a negative was predicted AND
        # the target is anything but a negative
        fn += np.logical_and(np.logical_not(pred_i), target_i).sum()
    # tp fp fn and tn are now single element tensors,
    # we must cast them back to ints
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)
    # pcpt stands for proportion_correctly_predicted_tokens
    pcpt = (tp + tn) / (tp + tn + fp + fn)
    return (*prec_rec_f1(tp, fp, fn), pcpt)


def pred_to_IOB2(pred, model, batch, label_mapping):
    """ Takes raw predictions from a NamedEntityRecognizer model of the form
        [[0.01, 0.9, 0.09], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], ...]
        where each sublist represents a probability distribution on entity
        types, and returns an IOB2-formatted prediction with no padding.
        Args:
            pred (Tensor): the model's predictions for the sequence
            model (NamedEntityRecognizer): the model being trained
            batch (DataLoader instance?): the batch being processed
            label_mapping (dict=None): maps label ids (indices in the
                probability distribution vector) to actual labels.
    """
    output = model.probabilities_to_prediction(
        model.output_to_probabilities(pred)).tolist()
    # Now we remove all of the padding elements from output & labels
    # so we do not skew the stats. For that, we need the true number
    # of tokens in each sequence
    seq_lens = [len(seq) for seq in batch.get("token_starts")]
    try:
        # labels = [seq[:length] for seq, length in zip(labels, seq_lens)]
        output = [seq[:length] for seq, length in zip(output, seq_lens)]
    except IndexError as e:
        print('batch.get("token_starts"):\n', batch.get("token_starts"),
              file=sys.stderr)
        print('seq_lens:\n', seq_lens, file=sys.stderr)
        # print('labels:\n', labels, file=sys.stderr)
        print('output:\n', output, file=sys.stderr)
        raise e

    # Currently, output_tracker contains label IDs instead of actual labels,
    # since that's what the model understands. Here, we're switching the IDs
    # with the labels themselves.
    output = [[label_mapping[tok] for tok in seq] for seq in output]
    return output


def evaluate(model, corpus, idx_to_labels, batch_size,
             collate_fn, write_pred=False):
    """ Evaluates a BELT model
        Args:
            - (TransformerModel) model: the model to evaluate
            - (MedMentionsCorpus) corpus: the evaluation corpus
            - (dict) idx_to_labels: a dict mapping token labels (UMLS
                CUIDs, STIDs etc.) to indices. Assumes all the CUIDs
                used in the corpus are indexed in idx_to_labels.
            - (int) batch_size: size of the batches to be processed
            - (func) collate_fn: function for corpus collation used
                by dataloader
            - (bool) write_pred: whether or not to write predictions to file.
                Defaults to False.
    """
    model.eval()  # Turn on the evaluation mode
    total_loss = []
    text_tagged = []
    text_targets = []
    corpus.load_instances()
    dataloader = DataLoader(
        corpus,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    with torch.no_grad():
        for batch in iter(dataloader):
            # here, the labels are the numbers as
            # mapped by the labels_to_idx dict
            labels = batch.get("token_labels")
            with autocast():
                raw_output = model(batch)
                loss = cst.criterion(raw_output,
                                     labels,
                                     batch.get("token_masks"))
                # loss = round(float(loss_func(raw_output, labels).data), 2)
            total_loss.append(float(loss))

            # we don't need the pre-mapped labels anymore, so we can just
            # overwrite the variable with the raw strings in nested lists
            labels = batch.get("raw_seq_labels")
            # In the same way, `output` contains labels as strings, not
            # as indices of the output vector of the classifer
            output = pred_to_IOB2(raw_output, model, batch, idx_to_labels)

            text_targets.extend(labels)
            text_tagged.extend(output)

    if write_pred:
        outfile = args['--predictions_fname']
        with open(outfile, 'wb') as f:
            pickle.dump(text_tagged, f)
        outfile = args['--targets_fname']
        with open(outfile, 'wb') as f:
            pickle.dump(text_targets, f)
    try:
        rep = classification_report(text_targets, text_tagged)
        f1 = f1_score(text_targets, text_tagged)
        prec = precision_score(text_targets, text_tagged)
        rec = recall_score(text_targets, text_tagged)
    except ValueError:
        rep = f1 = prec = None
        rec = "Error: could not compute Precision/Recall/F1. Too few classes."
    acc = accuracy_score(text_targets, text_tagged)
    return_val = mean(total_loss), acc, prec, rec, f1, rep
    return return_val


if __name__ == '__main__':
    args = get_evaluate_args(argv)

    with open(args['--test_fname'], 'rb') as test_file:
        test_corpus = pickle.load(test_file)
    with open(args['--model_fname'], 'rb') as model_file:
        best_model = torch.load(model_file)

    labels_to_idx = extract_label_mapping(args['--test_fname'])
    idx_to_labels = {v: k for k, v in labels_to_idx.items()}

    pad_id = best_model.tokenizer.pad_token_id

    loss, acc, prec, rec, f1, report =\
        evaluate(best_model, test_corpus, idx_to_labels, args['--batch_size'],
                 lambda b: collate_ner(b, pad_id=pad_id), args['--write_pred'])
