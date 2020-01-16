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

from evaluate import evaluate
from model import TransformerModel
from sys import argv


def train(model, corpus, umls_concepts, optimizer, scheduler, numericalizer,
          window_size=20, batch_size=35, epoch=0, log_interval=200):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for doc_idx, document in enumerate(corpus.documents()):
        # print(doc_idx, document.text[1:10])
        text = numericalizer.numericalize_text(document.text)
        targets = torch.zeros(batch_size,
                              # len(umls_concepts) + 1,
                              dtype=torch.long).to(device)
        data = torch.zeros(batch_size,
                           window_size,
                           dtype=torch.long).to(device)
        target_words = torch.zeros(batch_size, dtype=torch.long).to(device)
        for i in range(len(text)):
            start_index, end_index =\
                get_start_end_indices(i, len(text), window_size)
            data[i % batch_size] = get_text_window(
                text, window_size, start_index, end_index)

            target = umls_concepts[document.get_cuid(i)]
            targets[i % batch_size] = target
            target_words[i % batch_size] = text[i]

            # the fact that we do this processing only every batch_size
            # steps means that we don't do it if the remainder of the
            # text does not consitute a full batch. This is intentional,
            # and the standard way of handling this case.
            if i % batch_size == 0:
                optimizer.zero_grad()
                output = model(data, target_words)
                loss = criterion(output, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                targets = torch.zeros(batch_size,
                                      # len(umls_concepts) + 1,
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

    i = 1
    while i < len(argv) - 1:
        if argv[i] in args.keys():
            args[argv[i]] = argv[i + 1]
        i += 1
    args['--epochs'] = int(args['--epochs'])
    # try:
    #     _, train_fname, val_fname, model_fname = argv
    # except ValueError:
    #     print("Failed unpacking training corpus filename, "
    #           "validation corpus filename, and model destination "
    #           "filename from arguments. Proceeding with default values.")

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
    model = TransformerModel(ntoken=len(numericalizer.vocab),
                             n_umls_concepts=len(umls_concepts),
                             embed_size=200, nhead=2, nhid=200,
                             nlayers=2,  # batch_size=35,
                             phrase_len=20, dropout=0.2).to(device)
    print("running on: ", device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)  # lr=learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # start train
    best_val_loss = float("inf")
    best_model = None
    epochs_info = [["time", "train loss", "validation loss", "perplexity"]]

    for epoch in range(args['--epochs']):
        epoch_start_time = time.time()
        train(model, train_corpus, umls_concepts,
              optimizer, scheduler, numericalizer, epoch=epoch)

        train_loss = evaluate(model,
                              train_corpus,
                              umls_concepts,
                              numericalizer)
        val_loss = evaluate(model,
                            val_corpus,
                            umls_concepts,
                            numericalizer)

        print('-' * 89)
        try:
            valid_ppl = math.exp(val_loss)
        except OverflowError:
            print("validation ppl too large to compute")
            valid_ppl = "NA"
        current_epoch_info = [str(time.time() - epoch_start_time),
                              str(train_loss),
                              str(val_loss),
                              str(valid_ppl)]
        epochs_info.append(current_epoch_info)
        print(current_epoch_info)
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    with open(cst.train_stats_fname, 'w') as train_stats_file:
        writer = csv.csvwriter(train_stats_file, delimiter=';')
        writer.writerows(epochs_info)
    with open(args['--model_fname'], 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open(cst.numer_fname, 'wb') as numericalizer_file:
        pickle.dump(numericalizer, numericalizer_file)
