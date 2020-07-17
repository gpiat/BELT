import torch
import torch.nn as nn
import os

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

med_prefix = os.path.expanduser('~') + "/Documents/Datasets/MedMentions/"
full_corpus_fname = med_prefix + "st21pv/data/corpus_pubtator.txt"
med_corpus_train = med_prefix + "st21pv/data/corpus_pubtator_train.txt"
med_corpus_val = med_prefix + "st21pv/data/corpus_pubtator_val.txt"
med_corpus_test = med_prefix + "st21pv/data/corpus_pubtator_test.txt"
corpus_split_prefix = med_prefix + "full/data/"
train_corpus_pmids = corpus_split_prefix + "corpus_pubtator_pmids_trng.txt"
val_corpus_pmids = corpus_split_prefix + "corpus_pubtator_pmids_dev.txt"
test_corpus_pmids = corpus_split_prefix + "corpus_pubtator_pmids_test.txt"

pkl_prefix = os.path.dirname(os.path.realpath(__file__)) + "/pickles/"
umls_fname = pkl_prefix + "umls_concepts.pkl"
stid_fname = pkl_prefix + "semantic_types.pkl"
train_fname = pkl_prefix + "train.pkl"
val_fname = pkl_prefix + "val.pkl"
test_fname = pkl_prefix + "test.pkl"
numer_fname = pkl_prefix + "numericalizer.pkl"

wd = os.getcwd() + '/'
train_stats_fname = "/train_stats.csv"
model_fname = "model.pkl"
