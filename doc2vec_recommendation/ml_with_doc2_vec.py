import pandas as pd

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("../database/FinalTrainAbleDataset.xlsx").sample(frac=1)

df = df[['stop_clean_body','target']]


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


from gensim.models import Doc2Vec
model_dbow = Doc2Vec.load('C:\\Users\\Mehedi\\PycharmProjects\\NewsPaperSavedModel\\WithOutStopDoc2VecModel')

import numpy as np

def get_vectors(model,corpus_size,vectors_size,vectors_type):
    vectors = np.zeros((corpus_size,vectors_size))
    for i in range(0,corpus_size):
      prefix = vectors_type + '_' + str(i)
      vectors[i] = model.docvecs[prefix]
    return vectors

train_vectors_dbow = get_vectors(model_dbow,len(X_train), 300, "Train")
test_vectors_dbow = get_vectors(model_dbow,len(X_test), 300, "Test")



print('################### Doc2vec Logistic Regression ####################')
from sklearn.metrics import accuracy_score,classification_report
model = LogisticRegression()
model = model.fit(train_vectors_dbow, y_train)
pred = model.predict(test_vectors_dbow)
print('Validation accuracy {:.2f}%'.format(accuracy_score(pred, y_test)*100))
print(classification_report(y_test,pred))



print('################### Doc2vec SGD Classifier ####################')
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(max_iter=1000, tol=1e-3,n_jobs=2,penalty="l2")
model = model.fit(train_vectors_dbow, y_train)
pred = model.predict(test_vectors_dbow)
print('Validation accuracy {:.2f}%'.format(accuracy_score(pred, y_test)*100))
print(classification_report(y_test,pred))
