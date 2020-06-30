import constants as cst
import csv
import math
import pickle
import time
import torch

from constants import criterion
from constants import device

from util import Numericalizer
from util import get_start_end_indices
from util import get_text_window
from util import parse_args
from util import select_optimizer

from evaluate import evaluate
from model import TransformerModel
from sys import argv


def train(model, corpus, umls_concepts, optimizer, scheduler, numericalizer,
          batch_size, overlap=0.8, epoch=0, log_interval=200):
    """ Args:
            model
            corpus
            umls_concepts
            optimizer
            scheduler
            numericalizer: Numericalizer object for numericalizing text
            batch_size (int): number of batches of text to handle
                simultaneously
            overlap (float in [0,1]): proportion of the windows that should
                overlap
            epoch (int): for logging purposes, allows a custom start to the
                epoch counter when resuming training after an interruption
            log_interval (int): number of iterations between logging events
    """
    model.train()  # Turn on the train mode
    window_size = model.phrase_len
    total_loss = 0.
    start_time = time.time()
    for doc_idx, document in enumerate(corpus.documents()):
        # print(doc_idx, document.text[1:10])
        text = numericalizer.numericalize_text(document.text)
        targets = torch.zeros(batch_size,
                              window_size,
                              dtype=torch.long).to(device)
        data = torch.zeros(batch_size,
                           window_size,
                           dtype=torch.long).to(device)
        ## target_words = torch.zeros(batch_size, dtype=torch.long).to(device)

        # Here we're going over the text with a sliding window with overlap.
        # The idea is that the first quarter and last quarter of the predicted
        # labels likely don't have enough context to give an accurate
        # prediction.
        for i in range(0, len(text), round(((1 - overlap) / 2) * window_size)):
            start_index = min(i, len(text) - window_size)
            end_index = min(len(text), window_size + i)
            ## start_index, end_index = i, window_size + i
            # TODO: prevent processing the same text segment multiple times when
            # getting to the end of the text
            ## start_index, end_index = get_start_end_indices(i, len(text), window_size)

            data[i % batch_size] = get_text_window(
                text, window_size, start_index, end_index)

            target = [umls_concepts[cuid]
                      for cuid in document.get_cuids(start_index, end_index)]
            targets[i % batch_size] = torch.Tensor(target).to(device)
            ## target_words[i % batch_size] = text[i]

            # the fact that we do this processing only every batch_size
            # steps means that we don't do it if the remainder of the
            # text does not consitute a full batch. This is intentional,
            # and the standard way of handling this case.
            if i % batch_size == 0:
                optimizer.zero_grad()
                output = model(data)  # , target_words)
                loss = criterion(output, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                targets = torch.zeros(batch_size,
                                      window_size,
                                      dtype=torch.long).to(device)

        if doc_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            try:
                ppl = math.exp(cur_loss)
            except OverflowError:
                print("ppl too large to compute")
                ppl = 0
            print('| epoch {:3d} | {:5d}/{:5d} documents | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch,
                      doc_idx,
                      corpus.n_documents,
                      scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss,
                      ppl))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    args = {}
    args['--train_fname'] = cst.train_fname
    args['--val_fname'] = cst.val_fname
    args['--model_fname'] = cst.model_fname
    args['--epochs'] = 10
    args['--writepath'] = cst.wd
    args['--optim'] = "SGD"
    args['--lr'] = 5
    args['--window_size'] = 20
    args['--batch_size'] = 35

    parse_args(argv, args)
    args['--epochs'] = int(args['--epochs'])
    args['--lr'] = float(args['--lr'])
    args['--window_size'] = int(args['--window_size'])
    args['--batch_size'] = int(args['--batch_size'])

    try:
        with open(args['--train_fname'], 'rb') as train_file:
            train_corpus = pickle.load(train_file)
        with open(cst.umls_fname, 'rb') as umls_con_file:
            umls_concepts = pickle.load(umls_con_file)
        with open(args['--val_fname'], 'rb') as val_file:
            val_corpus = pickle.load(val_file)
    except pickle.UnpicklingError:
        print("Something went wrong when unpickling train, test, or "
              "validation corpus. Please ensure the files are valid "
              "python pickles.")
    except FileNotFoundError:
        print("One of the train, test, or validation corpus pickles "
              "was not found. Please ensure that the file specified "
              "as argument or in constants.py exists.")

    numericalizer = Numericalizer(train_corpus.vocab)
    with open(cst.numer_fname, 'wb') as numericalizer_file:
        pickle.dump(numericalizer, numericalizer_file)

    if '--resume' not in argv:
        model = TransformerModel(ntoken=len(numericalizer.vocab),
                                 n_umls_concepts=len(umls_concepts),
                                 embed_size=200, nhead=2, nhid=200,
                                 nlayers=2,  # batch_size=35,
                                 phrase_len=args['--window_size'],
                                 dropout=0.2).to(device)
    else:
        with open(args['--writepath'] + args['--model_fname'],
                  'rb') as model_file:
            model = pickle.load(model_file)

    print("running on: ", device)

    optimizer, scheduler = select_optimizer(
        option=args['--optim'].lower(), model=model, lr=args['--lr'])

    # start train
    best_val_loss = float("inf")
    best_model = None
    if '--resume' not in argv:
        column_headers = [["time", "train loss", "validation loss",
                           "perplexity", "train mention precision",
                           "train mention recall", "train mention F1",
                           "train document precision", "train document recall",
                           "train document F1", "val mention precision",
                           "val mention recall", "val mention F1",
                           "val document precision", "val document recall",
                           "val document F1"]]
        with open((args["--writepath"] +
                   cst.train_stats_fname), 'w') as train_stats_file:
            writer = csv.writer(train_stats_file, delimiter=';')
            writer.writerows(column_headers)

    for epoch in range(args['--epochs']):
        epoch_start_time = time.time()
        train(model, train_corpus, umls_concepts,
              optimizer, scheduler, numericalizer,
              batch_size=args["--batch_size"], epoch=epoch)

        (train_loss,
         train_mention_p_r_f1,
         train_doc_p_r_f1) = evaluate(model,
                                      train_corpus,
                                      umls_concepts,
                                      numericalizer,
                                      compute_p_r_f1=True)
        (val_loss,
         val_mention_p_r_f1,
         val_doc_p_r_f1) = evaluate(model,
                                    val_corpus,
                                    umls_concepts,
                                    numericalizer,
                                    compute_p_r_f1=True)
        val_corpus.loop_documents()

        print('-' * 89)
        try:
            valid_ppl = math.exp(val_loss)
        except OverflowError:
            print("validation ppl too large to compute")
            valid_ppl = "NA"
        current_epoch_info = [str(time.time() - epoch_start_time),
                              str(train_loss),
                              str(val_loss),
                              str(valid_ppl),
                              *train_mention_p_r_f1,
                              *train_doc_p_r_f1,
                              *val_mention_p_r_f1,
                              *val_doc_p_r_f1]
        print(current_epoch_info)
        print('-' * 89)
        # write epoch info at every epoch
        with open((args["--writepath"] +
                   cst.train_stats_fname), 'a') as train_stats_file:
            writer = csv.writer(train_stats_file, delimiter=';')
            writer.writerow(current_epoch_info)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            with open(args['--writepath'] +
                      args['--model_fname'], 'wb') as model_file:
                pickle.dump(best_model, model_file)

        scheduler.step()
