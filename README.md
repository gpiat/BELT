# BELT
Biomedical Entity Linking with Transformers

Much of the structure of this project is based on the pytorch tutorial ["Sequence-to-Sequence Modeling with nn.Transformer and TorchText"](https://pytorch.org/tutorials/beginner/transformer_tutorial.html).

The idea is to use a basic Transformer for Entity Linking in the UMLS biomedical knowledgebase. For this, we use the [MedMentions](https://github.com/chanzuckerberg/MedMentions/tree/master) corpus.

Results were very poor and never published. See [Samuel Broscheit's](https://arxiv.org/pdf/2003.05473.pdf) and [Chen _et al._'s](https://arxiv.org/abs/1911.03834) articles for basically the same approach, except using pretrained BERT and also it works(ish) on a subset of Wikipedia (which is much smaller than UMLS and has way more training data than MedMentions).

The reasons BELT didn't work were:
- Model was too small due to compute constraints
- We didn't pretrain as it would have taken precious compute time and we didn't think it would be as helpful as training on the actual objective. With the size of the model, we didn't think it would get very good at general NLU, and all we really wanted was fuzzy string matching and basic part of speech understanding to compete with string-matching-based tools.
- This approach fundamentally doesn't scale to 6 million classes (3 million concepts in UMLS, but tagged in IOB2 format) with a 6.6 MB training corpus (MedMentions)
