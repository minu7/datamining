import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from db import Sentence
from db import Keyword

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from train_utility import most_common_keywords
from train_utility import words_presences
from train_utility import words_counts

sentences = Sentence.find({ "type": "analyst" })
sentences = [sentence for sentence in sentences]

# lemmatizer = WordNetLemmatizer()
# lemma_sentences = [[lemmatizer.lemmatize(token) for token in word_tokenize(sentence["text"].lower())] for sentence in sentences]

pst = PorterStemmer()
lemma_sentences = [[pst.stem(token) for token in word_tokenize(sentence["text"].lower())] for sentence in sentences]

most_freq = most_common_keywords(lemma_sentences, 500)

X_presences_common = words_presences(most_freq, lemma_sentences)

y = [sentence["class"] for sentence in sentences]
# after import for test training: docker-compose run flask python app/api/train.py

"""
    SVM
"""
print("SVM WORD PRECENCES 500 MOST COMMON WORDS")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_presences_common, y, cv=10)
y_pred = cross_val_predict(clf, X_presences_common, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

"""
    KNN
"""

print("KNN WORD PRECENCES 500 MOST COMMON WORDS")
clf = knn(n_neighbors=3, metric='minkowski')
scores = cross_val_score(clf, X_presences_common, y, cv=10)
y_pred = cross_val_predict(clf, X_presences_common, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

"""
    SIMPLE LOGISTIC
"""

print("SIMPLE LOGISTIC WORD PRECENCES 500 MOST COMMON WORDS")
clf = lr()
scores = cross_val_score(clf, X_presences_common, y, cv=10)
y_pred = cross_val_predict(clf, X_presences_common, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')
