import constants as cst
import csv
import evaluate as ev
import math
import os
import time
import torch
import warnings

from args_handler import get_train_args
from args_handler import select_optimizer
from constants import criterion
from constants import device
from dataset import NERDataset
from dataset import collate_ner
from dataset import extract_label_mapping
from util import load_model

from math import mean
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from sys import argv
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


def train(model, corpus, target_finder, target_indexing, optimizer,
          scheduler, batch_size, scaler):
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
    dataloader = DataLoader(
        corpus,
        batch_size=batch_size,
        collate_fn=lambda b: collate_ner(b,
                                         pad_id=model.tokenizer.pad_token_id)
    )
    total_loss = []
    text_tagged = []
    text_targets = []

    for batch in iter(dataloader):
        optimizer.zero_grad()
        labels = batch.get("token_labels")
        with autocast():
            output = model(batch)
            loss = criterion(output, labels, batch.get("token_masks"))
            total_loss.append(float(loss))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        text_targets.extend(labels)
        text_tagged.extend(output)

    try:
        prec = precision_score(text_targets, text_tagged)
        rec = recall_score(text_targets, text_tagged)
        f1 = f1_score(text_targets, text_tagged)
    except ValueError:
        prec = rec = f1 = "Error"
    return_val = round(mean(total_loss), 2), prec, rec, f1
    return return_val


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


def write_results_to_file(current_epoch_info, train_stats_fname):
    # write epoch info at every epoch
    with open(train_stats_fname, 'a') as train_stats_file:
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

    dataset_files = {
        'train': args['--train_fname'],
        'dev': args['--dev_fname'],
        'test': args['--test_fname']
    }

    bert_tokenizer = BertTokenizer.from_pretrained(args['--bert_dir'])
    label_mapping = extract_label_mapping(file_list=dataset_files)

    train_corpus = NERDataset(medmentions_file=args.get("--train_fname"),
                              bert_tokenizer=bert_tokenizer,
                              label_mapping=label_mapping)
    dev_corpus = NERDataset(medmentions_file=args.get("--dev_fname"),
                            bert_tokenizer=bert_tokenizer,
                            label_mapping=label_mapping)

    model = load_model(args,
                       target_indexing=label_mapping,
                       tokenizer=bert_tokenizer)
    pad_id = model.tokenizer.pad_token_id

    print("running on: ", device)

    n_batches = math.ceil(len(train_corpus) / args["--batch_size"])
    optimizer, scheduler = select_optimizer(
        option=args['--optim'].lower(), model=model, lr=args['--lr'],
        n_batches=n_batches, epochs=args['--epochs'])

    # start train
    best_loss = float("inf")
    with open(os.path.join(args['--writepath'],
                           args['--model_fname']), 'wb') as model_file:
        # here pytorch warns us that it cannot perform sanity
        # checks on the model's source code, which we don't really
        # care about, so we ignore them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, model_file)
    best_model = None
    train_stats_fname = os.path.join(args["--writepath"],
                                     cst.train_stats_fname)
    if '--resume' not in argv:
        column_headers = [["time", "train loss", "dev loss",
                           "train precision", "train recall", "train F1",
                           "dev precision", "dev recall", "dev F1"]]
        with open(train_stats_fname, 'w') as train_stats_file:
            writer = csv.writer(train_stats_file, delimiter=';')
            writer.writerows(column_headers)

    scaler = GradScaler()
    for epoch in range(args['--epochs']):
        epoch_start_time = time.time()

        loss, p, r, f1 =\
            train(model, train_corpus, label_mapping, optimizer, scheduler,
                  batch_size=args["--batch_size"], scaler=scaler)
        epoch_time = time.time() - epoch_start_time
        epoch_info = tuple(str(i) for i in (p, r, f1))

        dev_loss, _, dev_p, dev_r, dev_f1, _ =\
            ev.evaluate(model, dev_corpus, label_mapping, args["--batch_size"],
                        collate_fn=lambda b: collate_ner(b, pad_id=pad_id))
        dev_prf1 = str(dev_p), str(dev_r), str(dev_f1)

        current_epoch_info = [str(epoch_time), str(loss), str(dev_loss),
                              *epoch_info, *dev_prf1]
        write_results_to_file(current_epoch_info, train_stats_fname)

        if loss < best_loss:
            best_loss = loss
            best_model = model
            write_model_to_file(best_model, args)

        print("End of epoch {}, took {} seconds.".format(epoch + 1,
                                                         epoch_time))

    # this runs main evaluation on test corpus. This function manages its
    # input corpus and output files autonomously.
    ev.main(args, best_model, bert_tokenizer, label_mapping)
