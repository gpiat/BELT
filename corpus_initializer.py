import constants as cst
import pickle

from sys import argv

from medmentions import MedMentionsCorpus


# def _full_vocab_initializer(corpus):
#     """
#     """
#     vocab = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
#     vocab = {word: (number + 4) for number, word in enumerate(corpus.vocab)}
#     return vocab

def _UMLS_concepts_initializer(corpus):
    """
    """
    umls_concepts = sorted(list(corpus.cuids))
    # we'll need to find the index given the CUID later so it's
    # easier to switch values and indices right away. We add 1 to
    # the index to reserve the zeroth index
    umls_concepts = {cuid: (index + 1)
                     for index, cuid in enumerate(umls_concepts)}
    # the zeroth concept is the non-concept.
    umls_concepts[None] = 0
    return umls_concepts


def UMLS_concepts_init(outfile=cst.umls_fname, corpora=None):
    """
    """
    if corpora is None:
        corpora = [cst.full_corpus_fname]
    corpus = MedMentionsCorpus(corpora)
    umls_concepts = _UMLS_concepts_initializer(corpus)
    with open(outfile, 'wb') as f:
        pickle.dump(umls_concepts, f)


def create_corpora():
    # creating sub-corpora for train, test and validation
    full_corpus = MedMentionsCorpus([cst.full_corpus_fname])
    n_train_docs = int(cst.train_proportion * full_corpus.n_documents)
    n_test_docs = int(cst.test_proportion * full_corpus.n_documents)
    current_file = cst.med_corpus_train
    for i, document in enumerate(full_corpus.documents()):
        # figuring out where we are in the corpus
        # to write to the appropriate file
        if i > (n_train_docs + n_test_docs):
            current_file = cst.med_corpus_val
        elif i > n_train_docs:
            current_file = cst.med_corpus_test
        document.write_to(current_file)


def pickle_corpora():
    train_corpus = MedMentionsCorpus([cst.med_corpus_train])
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
        UMLS_concepts_init()

    if no_valid_args:
        print("No valid arguments passed.\n"
              "To create the train, test and validation corpora "
              "from a larger corpus, use option --create.\n"
              "To pickle the train, test and validation corpora "
              "for use in the main script, use option --pickle.\n"
              "Remember to appropriately set your file paths in "
              "constants.py.\n")
