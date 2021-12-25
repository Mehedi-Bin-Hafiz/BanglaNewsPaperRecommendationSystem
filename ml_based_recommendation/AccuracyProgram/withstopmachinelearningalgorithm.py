
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_excel("/content/drive/MyDrive/colab/FinalDataset/FinalTrainAbleDataset.xlsx")
df.head(5)

x = df.clean_body.values.astype('U')
y = df['target'].values

print('######### Multi ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
MultiPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('Mulclf', MultinomialNB()) ])
MultiPipeLine.fit(X_train, y_train)
pred = MultiPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=20, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)
MultiPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('Mulclf', MultinomialNB()) ])
MultiPipeLine.fit(X_train, y_train)
pred = MultiPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=25, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=0)
MultiPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('Mulclf', MultinomialNB()) ])
MultiPipeLine.fit(X_train, y_train)
pred = MultiPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=30, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35, random_state=0)
MultiPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('Mulclf', MultinomialNB()) ])
MultiPipeLine.fit(X_train, y_train)
pred = MultiPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=35, accuracy = {0:.2f}".format(100*score),"%")

print('########### SVC ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
SVCPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('randomclf', SVC()) ])
SVCPipeLine.fit(X_train, y_train)
pred = SVCPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=20, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)
SVCPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SVC()) ])
SVCPipeLine.fit(X_train, y_train)
pred = SVCPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=25, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=0)
SVCPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SVC()) ])
SVCPipeLine.fit(X_train, y_train)
pred = SVCPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=30, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35, random_state=0)
SVCPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SVC()) ])
SVCPipeLine.fit(X_train, y_train)
pred = SVCPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=35, accuracy = {0:.2f}".format(100*score),"%")

print('########### KNN ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
KNNPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('randomclf', KNeighborsClassifier()) ])
KNNPipeLine.fit(X_train, y_train)
pred = KNNPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=20, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)
KNNPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', KNeighborsClassifier()) ])
KNNPipeLine.fit(X_train, y_train)
pred = KNNPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=25, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=0)
KNNPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', KNeighborsClassifier()) ])
KNNPipeLine.fit(X_train, y_train)
pred = KNNPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=30, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35, random_state=0)
KNNPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', KNeighborsClassifier()) ])
KNNPipeLine.fit(X_train, y_train)
pred = KNNPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=35, accuracy = {0:.2f}".format(100*score),"%")

print('########### SGD ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
SGDPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SGDClassifier()) ])
SGDPipeLine.fit(X_train, y_train)
pred = SGDPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=20, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)
SGDPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SGDClassifier()) ])
SGDPipeLine.fit(X_train, y_train)
pred = SGDPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=25, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=0)
SGDPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SGDClassifier()) ])
SGDPipeLine.fit(X_train, y_train)
pred = SGDPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=30, accuracy = {0:.2f}".format(100*score),"%")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35, random_state=0)
SGDPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
                  ('SGDclf', SGDClassifier()) ])
SGDPipeLine.fit(X_train, y_train)
pred = SGDPipeLine.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("test size=35, accuracy = {0:.2f}".format(100*score),"%")

#################### save the best model as joblib ######################

# import joblib
# X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
# SGDPipeLineFinal = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,5))),
#                   ('SGDclf', SGDClassifier()) ])
# SGDPipeLineFinal.fit(X_train, y_train)
# pred = SGDPipeLineFinal.predict(X_test)
# score=metrics.accuracy_score(y_test, pred)
# print("test size=20, accuracy = {0:.2f}".format(100*score),"%")
# joblib.dump(SGDPipeLineFinal,'/content/drive/MyDrive/colab/SavedModel/SGDbestMLModel')

