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
keywords = Keyword.find({ "type": "analyst" })
# keywords = list(set([lemmatizer.lemmatize(keyword["value"].lower().strip()) for keyword in keywords]))
keywords = list(set([pst.stem(keyword["value"].lower().strip()) for keyword in keywords]))

keywords.remove('')


X_presences_common = words_presences(most_freq, lemma_sentences)
X_counts_common = words_counts(most_freq, lemma_sentences)

X_presences_keywords = words_presences(keywords, lemma_sentences)
X_counts_keywords = words_counts(keywords, lemma_sentences)


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
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("SVM WORD COUNTS 500 MOST COMMON WORDS")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_counts_common, y, cv=10)
y_pred = cross_val_predict(clf, X_counts_common, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("SVM WORD PRECENCES KEYWORDS")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_presences_keywords, y, cv=10)
y_pred = cross_val_predict(clf, X_presences_keywords, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("SVM WORD COUNTS KEYWORDS")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_counts_keywords, y, cv=10)
y_pred = cross_val_predict(clf, X_counts_keywords, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
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

print("KNN WORD COUNTS 500 MOST COMMON WORDS")
clf = knn(n_neighbors=3, metric='minkowski')
scores = cross_val_score(clf, X_counts_common, y, cv=10)
y_pred = cross_val_predict(clf, X_counts_common, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("KNN WORD PRECENCES KEYWORDS")
clf = knn(n_neighbors=5, metric='minkowski')
scores = cross_val_score(clf, X_presences_keywords, y, cv=10)
y_pred = cross_val_predict(clf, X_presences_keywords, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("KNN WORD COUNTS KEYWORDS")
clf = knn(n_neighbors=3, metric='minkowski')
scores = cross_val_score(clf, X_counts_keywords, y, cv=10)
y_pred = cross_val_predict(clf, X_counts_keywords, y, cv=10)
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

print("SIMPLE LOGISTIC WORD COUNTS 500 MOST COMMON WORDS")
clf = lr()
scores = cross_val_score(clf, X_counts_common, y, cv=10)
y_pred = cross_val_predict(clf, X_counts_common, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("SIMPLE LOGISTIC WORD PRECENCES KEYWORDS")
clf = lr()
scores = cross_val_score(clf, X_presences_keywords, y, cv=10)
y_pred = cross_val_predict(clf, X_presences_keywords, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')

print("SIMPLE LOGISTIC WORD COUNTS KEYWORDS")
clf = lr()
scores = cross_val_score(clf, X_counts_keywords, y, cv=10)
y_pred = cross_val_predict(clf, X_counts_keywords, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(f'-1 = {np.count_nonzero(y_pred == -1)} 0 = {np.count_nonzero(y_pred == 0)} 1 = {np.count_nonzero(y_pred == 1)}')
