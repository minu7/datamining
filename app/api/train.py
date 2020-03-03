import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from db import Sentence
from db import Keyword

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from utility import most_common_keywords
from utility import words_presences
from utility import words_counts
from utility import transformer_embeddings
from sklearn.preprocessing import normalize

sentences = Sentence.find({ "type": "analyst" })
sentences = [sentence for sentence in sentences]
print(len(sentences))

X = transformer_embeddings([sentence['text'] for sentence in sentences])
y = [sentence["class"] for sentence in sentences]
# after import for test training: docker-compose run flask python app/api/train.py
print(y)
"""
    SVM
"""
print("SVM WORD PRECENCES 500 MOST COMMON WORDS")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=10)
y_pred = cross_val_predict(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf = clf.fit(X, y)
conf_mat = confusion_matrix(y, y_pred)
f_measure = f1_score(y, y_pred, average=None)
print(conf_mat.tolist())
print(accuracy_score(y, clf.predict(X)))

"""
    KNN
"""

print("KNN WORD PRECENCES 500 MOST COMMON WORDS")
clf = knn(n_neighbors=5, metric='minkowski')
X_normalized = normalize(X, norm='l1')
scores = cross_val_score(clf, X_normalized, y, cv=10)
y_pred = cross_val_predict(clf, X_normalized, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf = clf.fit(X_normalized, y)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat.tolist())
print(accuracy_score(y, clf.predict(X_normalized)))

"""
    SIMPLE LOGISTIC
"""

print("SIMPLE LOGISTIC WORD PRECENCES 500 MOST COMMON WORDS")
clf = lr()
scores = cross_val_score(clf, X, y, cv=10)
y_pred = cross_val_predict(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf = clf.fit(X, y)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat.tolist())
print(accuracy_score(y, clf.predict(X)))
