import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

med_prefix = "/home/gpiat/Documents/Datasets/MedMentions/"
full_corpus_fname = med_prefix + "st21pv/data/corpus_pubtator.txt"
med_corpus_train = med_prefix + "st21pv/data/corpus_pubtator_train.txt"
med_corpus_val = med_prefix + "st21pv/data/corpus_pubtator_val.txt"
med_corpus_test = med_prefix + "st21pv/data/corpus_pubtator_test.txt"

# train, test and validation corpora don't exist
# by default, this sets the proportions of each.
train_proportion = 2 / 3
test_proportion = 1 / 6
assert (train_proportion + test_proportion < 1)

pkl_prefix = "/home/gpiat/Documents/projects/entity_linking_transformers/pickles/"
umls_fname = pkl_prefix + "umls_concepts.pkl"
train_fname = pkl_prefix + "train.pkl"
val_fname = pkl_prefix + "val.pkl"
test_fname = pkl_prefix + "test.pkl"
model_fname = pkl_prefix + "model.pkl"
numer_fname = pkl_prefix + "numericalizer.pkl"
