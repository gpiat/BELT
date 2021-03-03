import constants as cst
import math
import os
import sys
import torch
import torch_optimizer

from pubtatortool.tokenization import TokenType
from torch.optim.lr_scheduler import StepLR
from transformers import get_linear_schedule_with_warmup


def parse_args(argv, args):
    """ parses a list of arguments into a dictionary
        more easily interpretable in code.
        Args:
            argv (list<str>): a list of arguments
            args (dct<str: any>): a dict
    """
    i = 1
    while i < len(argv):
        if argv[i] in args.keys():
            if isinstance(args[argv[i]], bool):
                # this is for cases where the user can add a "run mode"
                # argument such as --debug which changes the default
                # behavior. In this case, args['--debug'] would be set
                # to False by default, but calling `script --debug`
                # would then set args['--debug'] to True.
                args[argv[i]] = not args[argv[i]]
            else:
                # getting the type of the argument
                # so we can cast the string later
                arg_type_cast = type(args[argv[i]])
                # non boolean arguments of the form --window_size
                # are necessarily followed by a value.
                # `script --window_size 25` => args['--window_size']: 25
                try:
                    args[argv[i]] = arg_type_cast(argv[i + 1])
                except IndexError:
                    print("Error: no value specified for ", argv[i])
                    print("Attempting to continue with default value")
                except (TypeError, ValueError) as e:
                    print("warning: got error while processing arguments:")
                    print(e)
                    print("assuming this is handled in"
                          " appropriate get_*_args method")
                    args[argv[i]] = argv[i + 1]
        i += 1


def get_train_args(argv):
    args = {
        '--writepath': cst.out_dir,
        '--out_dir_suffix': '',

        '--train_fname': cst.train_fname,
        '--dev_fname': cst.dev_fname,
        '--test_fname': cst.test_fname,
        '--model_fname': cst.model_fname,
        '--bert_dir': cst.wd + '/data/biobert-v1.1',

        '--resume': False,
        '--epochs': 10,
        '--optim': "adamw",
        '--lr': None,  # only useful for SGD
        '--window_size': 128,
        '--batch_size': 8,
        '--embed_size': 200,  # size of embeddings
        '--nhead': 8,  # number of attention heads
        '--nhid': 250,  # dimension of the FFNN in the encoder
        '--nlayers': 4,  # number of `TransformerEncoderLayer`s in the encoder
        '--dropout': 0.1  # dropout value for `TransformerEncoderLayer`s
    }

    parse_args(argv, args)
    # Commented out the following because I realized I could
    #    probably automate it thanks to everything in python
    #    being an object, even functions and object types.
    # args['--epochs'] = int(args['--epochs'])
    # args['--lr'] = float(args['--lr'])
    # args['--window_size'] = int(args['--window_size'])
    # args['--batch_size'] = int(args['--batch_size'])
    # args['--overlap'] = float(args['--overlap'])

    if args['--target_type'] not in ["bin", "semtype", "cuid"]:
        help(args, "Invalid target type")
        sys.exit(1)

    args.update(get_evaluate_args(argv))
    return args


def get_corpus_init_args(argv):
    arglist = [
        # create_corpora specific
        ("--full_corpus_fname", cst.full_corpus_fname,
            ""),
        ("--train_corpus_pmids", cst.train_corpus_pmids,
            ""),
        ("--val_corpus_pmids", cst.val_corpus_pmids,
            ""),
        ("--test_corpus_pmids", cst.test_corpus_pmids,
            ""),

        ("--med_corpus_train", cst.med_corpus_train,
            "Path to PubTator file containing train corpus"),
        ("--med_corpus_dev", cst.med_corpus_dev,
            "Path to PubTator file containing dev corpus"),
        ("--med_corpus_test", cst.med_corpus_test,
            "Path to PubTator file containing test corpus"),

        # pickle_corpora specific
        ("--train_fname", cst.train_fname,
            "Path to file in which train corpus object should be serialized"),
        ("--dev_fname", cst.dev_fname,
            "Path to file in which dev corpus object should be serialized"),
        ("--test_fname", cst.test_fname,
            "Path to file in which test corpus object should be serialized"),
        ('--tokenization', TokenType.CHAR,
            "Type of tokenization: 'word', 'wordpiece' or 'character'."),

        # UMLS_concepts_init specific
        ("--umls_fname", cst.umls_fname,
            ""),
        ("--st21_fname", cst.stid_fname,
            ""),

        ("--vocab_file", cst.vocab_file,
            "Path to a file containing a vocabulary or "
            "name of a huggingface BERT-type model.")
    ]
    args = {k: v for k, v, _ in arglist}
    descriptions = {k: d for k, _, d in arglist}
    parse_args(argv, args)
    if type(args['--tokenization']) != TokenType:
        args['--tokenization'] = TokenType.from_str(args['--tokenization'])
    return args, descriptions


def get_evaluate_args(argv):
    args = {
        '--test_fname': cst.test_fname,

        '--model_fname': cst.model_fname,
        '--numer_fname': cst.numer_fname,

        '--umls_fname': cst.umls_fname,
        '--st21_fname': cst.stid_fname,

        '--writepath': cst.out_dir,
        '--out_dir_suffix': '',
        '--report_fname': "stats.out",
        '--predictions_fname': "predictions.out",
        '--targets_fname': "targets.out",

        # target type can be "bin" for pure entity identification,
        # "semtype" for semantic type IDs
        # or "cuid" for UMLS Concept Unique Identifiers
        '--target_type': 'cuid',

        '--write_pred': False,
        '--skip_eval': False,
        '--overlap': 0.2
    }
    parse_args(argv, args)

    args['--writepath'] = args['--writepath'] + args['--out_dir_suffix']
    del args['--out_dir_suffix']
    if not os.path.exists(args['--writepath']):
        os.makedirs(args['--writepath'])
    args['--predictions_fname'] = os.path.join(args['--writepath'],
                                               args['--predictions_fname'])
    args['--targets_fname'] = os.path.join(args['--writepath'],
                                           args['--targets_fname'])
    args['--report_fname'] = os.path.join(args['--writepath'],
                                          args['--report_fname'])

    return args


def select_optimizer(option, model, lr, n_batches=None, epochs=None):
    # optimizer selection

    if option == "adam":
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "radam":
        lr = 0.001 if lr is None else lr
        optimizer = torch_optimizer.RAdam(model.parameters(),
                                          lr=lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=0)
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adamw":
        lr = 1e-5 if lr is None else lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        max_steps = n_batches * epochs
        num_warmup_step = math.floor(max_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, round(num_warmup_step), max_steps
        )
    elif option == "adamax":
        optimizer = torch.optim.Adamax(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adabound":
        optimizer = torch_optimizer.AdaBound(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            final_lr=0.1,
            gamma=1e-3,
            eps=1e-8,
            weight_decay=0,
            amsbound=False,
        )
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "sparseadam":
        optimizer = torch.optim.SparseAdam(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "rprop":
        optimizer = torch.optim.Rprop(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "lamb":
        lr = 1e-3 if lr is None else lr
        optimizer = torch_optimizer.Lamb(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        scheduler = StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "asgd":
        optimizer = torch.optim.ASGD(model.parameters())
        scheduler = StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "accsgd":
        lr = 1e-3 if lr is None else lr
        optimizer = torch_optimizer.AccSGD(
            model.parameters(),
            lr=lr,
            kappa=1000.0,
            xi=10.0,
            small_const=0.7,
            weight_decay=0
        )
        scheduler = StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "sgdw":
        lr = 1e-3 if lr is None else lr
        optimizer = torch_optimizer.SGDW(
            model.parameters(),
            lr=lr,
            momentum=0,
            dampening=0,
            weight_decay=1e-2,
            nesterov=False,
        )
        scheduler = StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "sgd":
        lr = 5 if lr is None else lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer,
                           10.0,
                           gamma=0.01)
    else:
        raise TypeError("optimizer name not recognized")
    return optimizer, scheduler
