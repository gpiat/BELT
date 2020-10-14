import constants as cst
import csv

from medmentions import MedMentionsCorpus


full_corpus = MedMentionsCorpus([cst.full_corpus_fname], no_punct=True)
vocab = {}
umls_mentions = {}
for doc in full_corpus.documents():
    for word in doc.text:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
    for entity in doc.umls_entities:
        if entity.concept_ID in umls_mentions:
            umls_mentions[entity.concept_ID] += 1
        else:
            umls_mentions[entity.concept_ID] = 1

with open('stats/vocab.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile,
                        delimiter=' ',
                        dialect='unix',
                        quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
    for word in vocab.items():
        writer.writerow(list(word))

with open('stats/umls_mentions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile,
                        delimiter=' ',
                        dialect='unix',
                        quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
    for concept in umls_mentions.items():
        writer.writerow(list(concept))
