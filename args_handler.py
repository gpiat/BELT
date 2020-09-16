import constants as cst
import sys
import torch
import torch_optimizer

from util import TokenType


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
                # non boolean arguments of the form --window_size
                # are necessarily followed by a value.
                # `script --window_size 25` => args['--window_size']: 25
                try:
                    args[argv[i]] = argv[i + 1]
                except IndexError:
                    print("Error: no value specified for ", argv[i])
        i += 1


def get_train_args(argv):
    args = {
        '--train_fname': cst.train_fname,
        '--val_fname': cst.val_fname,
        '--model_fname': cst.model_fname,
        '--writepath': cst.wd,
        '--resume': False,
        '--epochs': 10,
        '--optim': "SGD",
        '--lr': 5,
        '--window_size': 20,
        '--batch_size': 35,
        '--overlap': 0.2,
        # target type can be "bin" for pure entity identification,
        # "semtype" for semantic type IDs
        # or "cuid" for UMLS Concept Unique Identifiers
        '--target_type': 'cuid'
    }

    parse_args(argv, args)
    args['--epochs'] = int(args['--epochs'])
    args['--lr'] = float(args['--lr'])
    args['--window_size'] = int(args['--window_size'])
    args['--batch_size'] = int(args['--batch_size'])
    args['--overlap'] = float(args['--overlap'])

    if args['--target_type'] not in ["bin", "semtype", "cuid"]:
        help(args, "Invalid target type")
        sys.exit(1)

    return args


def get_corpus_init_args(argv):
    args = {
        "--full_corpus_fname": cst.full_corpus_fname,
        "--train_corpus_pmids": cst.train_corpus_pmids,
        "--val_corpus_pmids": cst.val_corpus_pmids,
        "--test_corpus_pmids": cst.test_corpus_pmids,

        # create_corpora specific
        "--med_corpus_train": cst.med_corpus_train,
        "--med_corpus_val": cst.med_corpus_val,
        "--med_corpus_test": cst.med_corpus_test,

        # pickle_corpora specific
        "--train_fname": cst.train_fname,
        "--val_fname": cst.val_fname,
        "--test_fname": cst.test_fname,
        "--nopunct": False,
        "--split_by_char": False,

        # UMLS_concepts_init specific
        "--umls_fname": cst.umls_fname,
        "--st21_fname": cst.stid_fname,

        '--tokenization': TokenType.CHAR
    }
    parse_args(argv, args)
    if type(args['--tokenization']) != TokenType:
        args['--tokenization'] = TokenType.form_str(args['--tokenization'])
    return args


def get_evaluate_args(argv):
    args = {
        '--test_fname': cst.test_fname,

        '--model_fname': cst.model_fname,
        '--numer_fname': cst.numer_fname,

        '--umls_fname': cst.umls_fname,
        "--st21_fname": cst.stid_fname,

        '--predictions_fname': cst.wd + "predictions.out",
        '--targets_fname': cst.wd + "targets.out",

        # target type can be "bin" for pure entity identification,
        # "semtype" for semantic type IDs
        # or "cuid" for UMLS Concept Unique Identifiers
        '--target_type': 'cuid',

        '--write_pred': False,
        '--skip_eval': False,
        '--overlap': 0.2
    }
    parse_args(argv, args)
    args['--overlap'] = float(args['--overlap'])
    return args


def select_optimizer(option, model, lr):
    # optimizer selection

    if option == "adam":
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(),
                                          lr=0.001,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adamax":
        optimizer = torch.optim.Adamax(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "sparseadam":
        optimizer = torch.optim.SparseAdam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "rprop":
        optimizer = torch.optim.Rprop(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "lamb":
        optimizer = torch_optimizer.Lamb(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
    elif option == "asgd":
        optimizer = torch.optim.ASGD(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "accsgd":
        optimizer = torch_optimizer.AccSGD(
            model.parameters(),
            lr=1e-3,
            kappa=1000.0,
            xi=10.0,
            small_const=0.7,
            weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "sgdw":
        optimizer = torch_optimizer.SGDW(
            model.parameters(),
            lr=1e-3,
            momentum=0,
            dampening=0,
            weight_decay=1e-2,
            nesterov=False,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.2)
    elif option == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    10.0,
                                                    gamma=0.01)
    else:
        raise TypeError("optimizer name not recognized")
    return optimizer, scheduler
