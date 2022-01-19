
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

df = pd.read_excel("../../database/FinalTrainAbleDataset.xlsx")

df.head(5)
x = df.stop_clean_body.values.astype('U')
y = df['target'].values

print('######### Multi ############')

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
MultiPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,3))),
                  ('Mulclf', MultinomialNB()) ])
MultiPipeLine.fit(X_train, y_train)
y_pred = MultiPipeLine.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")




print('########### SVC ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
SVCPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,3))),
                  ('randomclf', SVC()) ])
SVCPipeLine.fit(X_train, y_train)
y_pred = SVCPipeLine.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")





print('########### KNN ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
KNNPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,3))),
                  ('randomclf', KNeighborsClassifier()) ])
KNNPipeLine.fit(X_train, y_train)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")


print('########### SGD ############')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
SGDPipeLine = Pipeline([('tfidf', TfidfVectorizer(analyzer='word',ngram_range=(1,3))),
                  ('SGDclf', SGDClassifier(max_iter=1000, tol=1e-3,n_jobs=2,penalty="l2")) ])
SGDPipeLine.fit(X_train, y_train)
y_pred = SGDPipeLine.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")

