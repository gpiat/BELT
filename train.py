import constants as cst
import csv
import math
import pickle
import sys
import time
import torch

from args_handler import get_train_args
from args_handler import select_optimizer

from constants import criterion
from constants import device

from util import Numericalizer
from util import get_text_window
from util import pad
from util import set_targets
from util import load_model

from evaluate import evaluate
from sys import argv


def train(model, corpus, target_finder, target_indexing, optimizer,
          scheduler, numericalizer, batch_size, overlap=0.2, epoch=0,
          log_interval=200):
    """ Args:
            model
            corpus
            target_finder: callable function that finds the target
                for the text span
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
        text = numericalizer.numericalize_text(pad(document.text,
                                                   window_size,
                                                   overlap,
                                                   batch_size))
        targets = torch.zeros(batch_size,
                              window_size,
                              dtype=torch.long).to(device)
        data = torch.zeros(batch_size,
                           window_size,
                           dtype=torch.long).to(device)

        # Here we're going over the text with a sliding window with overlap.
        # The idea is that the first x% and last x% of the predicted
        # labels likely don't have enough bidirectional context to give an
        # accurate prediction.
        increment = round((1 - (overlap / 2)) * window_size)
        # example: overlap = 0.2, window_size = 10
        # 1 - (overlap / 2) = 0.9
        # increment = 9
        # the use of `round` is a response to floating point errors.
        # Without it, `increment` might be something like 8.9999 in some cases.
        for i in range(0, len(text), increment):
            start_index = max(min(i, len(text) - window_size), 0)
            end_index = min(len(text), window_size + i)
            # TODO: prevent processing the same text segment multiple times
            # when getting to the end of the text.
            # UPDATE: what am I talking about? Why would the text segment be
            # processed multiple times??

            data[i % batch_size] = get_text_window(
                text, window_size, start_index, end_index)

            target = target_finder(document, start_index,
                                   end_index, target_indexing)
            targets[i % batch_size] = torch.Tensor(target).to(device)

            # the fact that we do this processing only every batch_size
            # steps means that we don't do it if the remainder of the
            # text does not constitute a full batch. This is intentional,
            # and the standard way of handling this case.
            # UPDATE: the case where the remainder of the text does not
            # constitute a full batch no longer occurs thanks to padding.
            if i % batch_size == 0:
                optimizer.zero_grad()
                output = model(data)
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


def help(args, issue_description=""):
    print(issue_description)
    arg_descriptions = [
        "preloaded pickled training corpus filename",
        "preloaded pickled validation / dev corpus filename",
        "filename in which to save the model and from which to"
        " load it if not training from scratch",
        "path for all files to be written (model, stat files)",
        "number of epochs to train for",
        "name of the optimizer",
        "learning rate",
        "size of the window of input tokens",
        "size of the batch of windows",
        "proportion of overlap between windows (in [0,1])",
        'can be "bin" for pure entity identification,'
        ' "semtype" for semantic type IDs'
        ' or "cuid" for UMLS Concept Unique Identifiers'
    ]
    for item in zip(args.keys(), arg_descriptions):
        print(item)


def load_files(args):
    try:
        with open(args['--train_fname'], 'rb') as train_file:
            train_corpus = pickle.load(train_file)
        with open(args['--val_fname'], 'rb') as val_file:
            dev_corpus = pickle.load(val_file)
        if ((args['--target_type'] == "cuid" or
             args['--target_type'] == "bin")):
            with open(cst.umls_fname, 'rb') as umls_con_file:
                target_indexing = pickle.load(umls_con_file)
        else:
            with open(cst.stid_fname, 'rb') as semtype_file:
                target_indexing = pickle.load(semtype_file)
    except pickle.UnpicklingError:
        print("Something went wrong when unpickling the train corpus, "
              "dev corpus, or UMLS concepts file. Please ensure the "
              "specified files are valid python pickles.")
        sys.exit(1)
    except FileNotFoundError:
        print("One of the train corpus, dev corpus, or UMLS concept "
              "pickles was not found. Please ensure that the file "
              "specified as argument or in constants.py exists.")
        sys.exit(1)
    return train_corpus, target_indexing, dev_corpus


if __name__ == '__main__':
    args = get_train_args(argv)
    target_finder = set_targets(args['--target_type'])

    train_corpus, target_indexing, dev_corpus = load_files(args)

    numericalizer = Numericalizer(train_corpus.vocab)
    with open(cst.numer_fname, 'wb') as numericalizer_file:
        pickle.dump(numericalizer, numericalizer_file)

    model = load_model(argv,
                       args,
                       vocab_size=len(numericalizer.vocab),
                       n_classes=len(target_indexing))

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
        # TODO: figure out why this loop never stops
        epoch_start_time = time.time()
        train(model, train_corpus, target_finder,
              target_indexing, optimizer, scheduler,
              numericalizer, batch_size=args["--batch_size"],
              overlap=args['--overlap'], epoch=epoch)

        (train_loss,
         train_mention_p_r_f1,
         train_doc_p_r_f1) = evaluate(model,
                                      train_corpus,
                                      target_finder,
                                      target_indexing,
                                      numericalizer,
                                      args['--overlap'],
                                      compute_p_r_f1=True)

        # TODO: figure out why ever since masking was implemented, model
        # never saves itself to file, implying val_loss never decreases.
        (val_loss,
         val_mention_p_r_f1,
         val_doc_p_r_f1) = evaluate(model,
                                    dev_corpus,
                                    target_finder,
                                    target_indexing,
                                    numericalizer,
                                    args['--overlap'],
                                    compute_p_r_f1=True)
        dev_corpus.loop_documents()

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
