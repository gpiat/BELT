import constants as cst
import sys


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
    args = {}
    args['--train_fname'] = cst.train_fname
    args['--val_fname'] = cst.val_fname
    args['--model_fname'] = cst.model_fname
    args['--writepath'] = cst.wd
    args['--epochs'] = 10
    args['--optim'] = "SGD"
    args['--lr'] = 5
    args['--window_size'] = 20
    args['--batch_size'] = 35
    args['--overlap'] = 0.2
    # target type can be "bin" for pure entity identification,
    # "semtype" for semantic type IDs
    # or "cuid" for UMLS Concept Unique Identifiers
    args['--target_type'] = 'cuid'

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
        "--st21_fname": cst.stid_fname
    }
    parse_args(argv, args)
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

        '--write_pred': False,
        '--skip_eval': False,
        '--overlap': 0.2
    }
    parse_args(argv, args)
    args['--overlap'] = float(args['--overlap'])
    return args
