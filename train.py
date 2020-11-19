import constants as cst
import csv
import math
import pickle
import sys
import time
import torch
import warnings

from args_handler import get_train_args
from args_handler import select_optimizer

from constants import criterion
from constants import device

from util import get_text_window
from util import pad
from util import set_targets
from util import load_model

from evaluate import evaluate
from sys import argv


def train(model, corpus, target_finder, target_indexing, optimizer,
          scheduler, batch_size, overlap=0.2, epoch=0,
          log_interval=200):
    """ Args:
            model
            corpus
            target_finder: callable function that finds the target
                for the text span
            optimizer
            scheduler
            batch_size (int): number of batches of text to handle
                simultaneously
            overlap (float in [0,1[): proportion of the windows that should
                overlap
            epoch (int): for logging purposes, allows a custom start to the
                epoch counter when resuming training after an interruption
            log_interval (int): number of iterations between logging events
    """
    model.train()  # Turn on the train mode

    # henceforth we refer to sequences as windows, as the
    # overlap feature makes it practical to think about
    # the sequences like a sliding window over the text.
    window_size = model.phrase_len
    total_loss = 0.
    start_time = time.time()

    # We will be going over the text with a sliding window, which
    # means the sequences will overlap. The idea is that the first
    # and last x% of the predicted labels for a given sequence likely
    # don't have enough bidirectional context to give an accurate
    # prediction. This `increment` is the number of tokens that we skip
    # to get the next window.
    increment = round((1 - (overlap / 2)) * window_size)
    # example: overlap = 0.2, window_size = 10
    # 1 - (overlap / 2) = 0.9
    # increment = 0.9 * window_size = 9
    # In this case, there is one token of overlap at the beginning of
    # the window and one at the end.
    # The use of `round` is a response to floating point errors.
    # Without it, `increment` might be something like 8.9999 in some cases.

    for doc_idx, document in enumerate(corpus.documents()):
        # Padding the text, this allows the text to be cut up
        # neatly into appropriately sized chunks to fill up a
        # round number of batches
        padded_text = pad(document.text, window_size, overlap,
                          batch_size=batch_size,
                          pad_token=model.tokenizer.pad_token)

        # here huggingface may tell us that the sequence is too long
        # to pass on to BERT as is. We do not care about this as
        # we cut the text into batches afterwards.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = model.tokenizer.encode(padded_text)

        # initializing targets and model inputs
        targets = torch.zeros(batch_size,
                              window_size,
                              dtype=torch.long).to(device)
        data = torch.zeros(batch_size,
                           window_size,
                           dtype=torch.long).to(device)

        i = 0
        while True:
            # For clarification: this is not an infinite loop, it's the
            # official python syntax for a do/while. Complain to Guido.

            start_index = i * increment
            end_index = i * increment + window_size
            if end_index > len(text):
                break
            # You may be wondering: "What if start_index ends up less than
            # `increment` tokens away from the end of the window? Couldn't
            # we miss some text if it doesn't neatly fit into a number of
            # batches?"
            # That can't happen, thanks to padding.

            def debug():
                """
                    This is a debugging function that is theoretically no
                    longer useful but this specific block of code has given
                    me a lot of trouble so I'm leaving it here just in case
                """
                print("data shape: {}".format(data.shape))
                print("window number {}, in position {} of batch with "
                      "size {}".format(i, i % batch_size, batch_size))
                print("expected window size: {}, start index: {},"
                      " end index: {}, got window_size: {}".format(
                          window_size, start_index, end_index,
                          end_index - start_index))

            # Every iteration of this loop just fills up the
            # `data` matrix and its corresponding targets
            # progressively until we fill up a batch
            try:
                tw = get_text_window(text, window_size, start_index, end_index,
                                     pad_token=model.tokenizer.pad_token_id)
            except RuntimeError as e:
                debug()
                raise e

            try:
                data[i % batch_size] = tw
            except RuntimeError as e:
                print("text window shape: {}".format(tw.shape))
                debug()
                print("text window:")
                print(tw)
                raise e

            target = target_finder(document, start_index,
                                   end_index, target_indexing)
            tensor_target = torch.Tensor(target).to(device)
            targets[i % batch_size][:tensor_target.size()[0]] = tensor_target

            # When the batch is full, actually process it
            if (i + 1) % batch_size == 0:
                optimizer.zero_grad()
                output = model(data)

                # Here too, the debugging code is theoretically no
                # longer useful but this specific block of code has
                # given me a lot of trouble
                try:
                    loss = criterion(output, targets)
                except Exception as e:
                    print("targets.dtype: {}".format(targets.dtype))
                    print("targets: {}".format(targets))
                    print("output.dtype: {}".format(output.dtype))
                    print("output: {}".format(output))
                    raise e
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                # Resetting targets and data
                targets = torch.zeros(batch_size,
                                      window_size,
                                      dtype=torch.long).to(device)
                data = torch.zeros(batch_size,
                                   window_size,
                                   dtype=torch.long).to(device)
            i += 1

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
            # this line meant that the time of each log interval was
            # measured on its own. That might've been what I wanted
            # to do at the time, but makes little sense now.
            # start_time = time.time()


def help(args, issue_description=""):
    print(issue_description)
    arg_descriptions = [
        "preloaded pickled training corpus filename",
        "preloaded pickled dev corpus filename",
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
        with open(args['--val_fname'], 'rb') as dev_file:
            dev_corpus = pickle.load(dev_file)
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


def evaluate_model_performance(model, train_corpus, target_finder,
                               target_indexing, args, dev_corpus,
                               epoch_start_time):
    (train_loss,
     train_mention_p_r_f1,
     train_doc_p_r_f1) = evaluate(model,
                                  train_corpus,
                                  target_finder,
                                  target_indexing,
                                  args['--overlap'],
                                  compute_mntn_p_r_f1=True,
                                  compute_doc_p_r_f1=True)

    (dev_loss,
     dev_mention_p_r_f1,
     dev_doc_p_r_f1) = evaluate(model,
                                dev_corpus,
                                target_finder,
                                target_indexing,
                                args['--overlap'],
                                compute_mntn_p_r_f1=True,
                                compute_doc_p_r_f1=True)

    # print('-' * 89)
    try:
        dev_ppl = math.exp(dev_loss)
    except OverflowError:
        # print("Dev perplexity too large to compute")
        dev_ppl = "NA"
    current_epoch_info = [str(time.time() - epoch_start_time),
                          str(train_loss),
                          str(dev_loss),
                          str(dev_ppl),
                          *train_mention_p_r_f1,
                          *train_doc_p_r_f1,
                          *dev_mention_p_r_f1,
                          *dev_doc_p_r_f1]
    # print(current_epoch_info)
    # print('-' * 89)
    return current_epoch_info, dev_loss


def write_results_to_file(current_epoch_info, args):
    # write epoch info at every epoch
    with open((args["--writepath"] +
               cst.train_stats_fname), 'a') as train_stats_file:
        writer = csv.writer(train_stats_file, delimiter=';')
        writer.writerow(current_epoch_info)


def write_model_to_file(model, args):
    with open(args['--writepath'] +
              args['--model_fname'], 'wb') as model_file:
        # here pytorch warns us that it cannot perform sanity
        # checks on the model's source code, which we don't really
        # care about, so we ignore them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, model_file)


if __name__ == '__main__':
    args = get_train_args(argv)
    target_finder = set_targets(args['--target_type'])

    train_corpus, target_indexing, dev_corpus = load_files(args)

    model = load_model(args,
                       target_indexing=target_indexing,
                       tokenizer=train_corpus.tokenizer)

    print("running on: ", device)

    optimizer, scheduler = select_optimizer(
        option=args['--optim'].lower(), model=model, lr=args['--lr'])

    # start train
    best_loss = float("inf")
    with open(args['--writepath'] +
              args['--model_fname'], 'wb') as model_file:
        # here pytorch warns us that it cannot perform sanity
        # checks on the model's source code, which we don't really
        # care about, so we ignore them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, model_file)
    best_model = None
    if '--resume' not in argv:
        column_headers = [["time", "train loss", "dev loss",
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

    start_time = time.time()
    start_info, loss =\
        evaluate_model_performance(model, train_corpus, target_finder,
                                   target_indexing, args, dev_corpus,
                                   start_time)
    write_results_to_file(start_info, args)

    for epoch in range(args['--epochs']):
        epoch_start_time = time.time()

        train(model, train_corpus, target_finder,
              target_indexing, optimizer, scheduler,
              batch_size=args["--batch_size"],
              overlap=args['--overlap'], epoch=epoch)

        current_epoch_info, loss =\
            evaluate_model_performance(model, train_corpus,
                                       target_finder,
                                       target_indexing,
                                       args, dev_corpus,
                                       epoch_start_time)
        write_results_to_file(current_epoch_info, args)

        if loss < best_loss:
            best_loss = loss
            best_model = model
            write_model_to_file(best_model, args)

        print("End of epoch {}, advancing scheduler".format(epoch + 1))
        scheduler.step()
