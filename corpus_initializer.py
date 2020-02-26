import constants as cst
import pickle

from sys import argv

from medmentions import MedMentionsCorpus


def _UMLS_concepts_initializer(corpus, dct):
    """ Given a MedMentionsCorpus-like object as argument, which contains a
        collection of CUIDs, returns a dictionary mapping CUIDs to indices.
    """
    # we'll need to find the index given the CUID later so it's
    # easier to switch values and indices right away. We add 2 to
    # the index to reserve the zeroth and first indices
    umls_concepts = {cuid: (index + 2 if number > dct else 1)
                     for index, (cuid, number) in
                     enumerate(corpus.cuids.items())}

    # TODO: erase this old code from before the
    #   full default category implementation
    # umls_concepts = {cuid: (index + 2)
    #                  for index, cuid in enumerate(umls_concepts)}
    # umls_concepts["other"] = 1

    # the zeroth concept is the non-concept.
    umls_concepts[None] = 0
    return umls_concepts


def UMLS_concepts_init(outfile=cst.umls_fname,
                       corpora=None, dflt_cat_thresh=0):
    """ Pickles a dictionary mapping CUIDs found in a list of corpora to indices.
        Args:
            - (str) outfile: path of the file to write the pickle to.
                Defaults to the file name given in constants.py.
            - (list<str>) corpora: filenames of the PubTator-format
                corpora to use. All the CUIDs used come from these corpora.
            - (int) dflt_cat_thresh: any CUID that appears fewer times than
                this threshold is put in a "default" category.
    """
    if corpora is None:
        corpora = [cst.full_corpus_fname]
    corpus = MedMentionsCorpus(corpora)
    umls_concepts = _UMLS_concepts_initializer(corpus)
    with open(outfile, 'wb') as f:
        pickle.dump(umls_concepts, f)


def create_corpora():
    """ Splits the full corpus given in constants.py into training, validation,
        and test corpora. The proportions of documents going into each corpus
        are set in constants.py as well. These smaller corpora are written to
        the disk (destination file names are set in constants.py) in plaintext
        PubTator format.
    """
    # creating sub-corpora for train, test and validation
    full_corpus = MedMentionsCorpus([cst.full_corpus_fname])

    pmids = {}
    with open(cst.train_corpus_pmids, 'r') as f:
        pmids['train'] = [pmid.strip() for pmid in f.readlines()]
    with open(cst.val_corpus_pmids, 'r') as f:
        pmids['val'] = [pmid.strip() for pmid in f.readlines()]
    with open(cst.test_corpus_pmids, 'r') as f:
        pmids['test'] = [pmid.strip() for pmid in f.readlines()]

    for document in full_corpus.documents():
        if document.pmid in pmids['train']:
            document.write_to(cst.med_corpus_train)
        elif document.pmid in pmids['val']:
            document.write_to(cst.med_corpus_val)
        elif document.pmid in pmids['test']:
            document.write_to(cst.med_corpus_test)


def pickle_corpora():
    """ Creates MedMentionsCorpus objects based on the train, validation and
        test corpora whose filenames are set in constants.py, which are then
        pickled. This avoids object creation overhead (which is fairly
        expensive) when running many experiments on the same corpus.
    """
    train_corpus = MedMentionsCorpus([cst.med_corpus_train], auto_looping=True)
    val_corpus = MedMentionsCorpus([cst.med_corpus_val])
    test_corpus = MedMentionsCorpus([cst.med_corpus_test])

    with open(cst.train_fname, 'wb') as train_file:
        pickle.dump(train_corpus, train_file)
    with open(cst.val_fname, 'wb') as val_file:
        pickle.dump(val_corpus, val_file)
    with open(cst.test_fname, 'wb') as test_file:
        pickle.dump(test_corpus, test_file)


if __name__ == '__main__':
    no_valid_args = True

    if "--create" in argv:
        no_valid_args = False
        create_corpora()

    if "--pickle" in argv:
        no_valid_args = False
        pickle_corpora()
        dct = 0
        if "--dct" in argv:
            # number that (hopefully) comes after "--dct" in argv
            dct = int(argv[argv.index["--dct"] + 1])
        UMLS_concepts_init(dflt_cat_thresh=dct)

    if no_valid_args:
        print("No valid arguments passed.\n"
              "To create the train, test and validation corpora "
              "from a larger corpus, use option --create.\n"
              "To pickle the train, test and validation corpora "
              "for use in the main script, use option --pickle.\n"
              "Remember to appropriately set your file paths in "
              "constants.py.\n")
