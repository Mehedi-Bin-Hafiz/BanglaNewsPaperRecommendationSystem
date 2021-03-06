# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("/content/drive/MyDrive/colab/FinalDataset/FinalTrainAbleDataset.xlsx").sample(frac=1)


df.head(2)

df = df[['stop_clean_body','target']]

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def label_sentences(corpus, label_type):
  labeled = []
  for i,v in enumerate(corpus):
    label = label_type + '_' +str(i)
    labeled.append(TaggedDocument(v.split(),[label]))
  return labeled

X_train, X_test, y_train, y_test = train_test_split(df.stop_clean_body, df.target, random_state= 0 , test_size = 0.2)

print(X_train.shape, X_test.shape)

X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')

len(X_train )

all_data = X_train + X_test

#################### Doc2VecDesign #####################
import multiprocessing
cores = multiprocessing.cpu_count()
from sklearn import utils
model_dbow = Doc2Vec(dm=0,vector_size = 300, negative = 5, min_count =1, alpha = 0.065, min_alpha = 0.065,workers=cores)
model_dbow.build_vocab([x for x in tqdm(all_data)])
for epoch in range(50):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

from gensim.test.utils import get_tmpfile
fname = get_tmpfile("/content/drive/MyDrive/colab/SavedModel/WithOutStopDoc2VecModel")
model_dbow.save(fname)

from gensim.models import Doc2Vec
model_dbow = Doc2Vec.load('/content/drive/MyDrive/colab/SavedModel/WithOutStopDoc2VecModel')

import numpy as np

def get_vectors(model,corpus_size,vectors_size,vectors_type):
    vectors = np.zeros((corpus_size,vectors_size))
    for i in range(0,corpus_size):
      prefix = vectors_type + '_' + str(i)
      vectors[i] = model.docvecs[prefix]
    return vectors

train_vectors_dbow = get_vectors(model_dbow,len(X_train), 300, "Train")
test_vectors_dbow = get_vectors(model_dbow,len(X_test), 300, "Test")

print(len(X_train))

