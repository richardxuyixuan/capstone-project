"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.

In this example, we reduce the dimensionality of the embeddings to e.g. 128 dimensions. This significantly
reduces the required memory / storage while maintaining nearly the same performance.

For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.

PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed.
"""
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util, evaluation, models, InputExample
import os
import gzip
import csv
import random
import numpy as np
import torch

model = SentenceTransformer('all-mpnet-base-v2')
new_dimension = 128

nli_dataset_path = 'datasets/AllNLI.tsv.gz'
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

eval_examples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            eval_examples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# Evaluate the original model on the STS benchmark dataset
stsb_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(eval_examples, name='sts-benchmark-test')
######## Reduce the embedding dimensions ########

#Read sentences from NLI dataset
nli_sentences = set()
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        nli_sentences.add(row['sentence1'])
        nli_sentences.add(row['sentence2'])

nli_sentences = list(nli_sentences)
random.shuffle(nli_sentences)

pca_train_sentences = nli_sentences[0:20000]
train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

#Compute PCA on the train embeddings matrix
pca = PCA(n_components=new_dimension)
pca.fit(train_embeddings)
pca_comp = np.asarray(pca.components_)

# We add a dense layer to the model, so that it will produce directly embeddings with the new size
dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
model.add_module('dense', dense)

model.save('models/my-128dim-model')

# You can then load the adapted model that produces 128 dimensional embeddings like this:
#model = SentenceTransformer('models/my-128dim-model')
