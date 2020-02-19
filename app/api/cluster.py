import numpy as np

from db import Sentence
from db import Keyword

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from train_utility import most_common_keywords
from train_utility import words_presences
import xlrd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import io
import base64

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def cluster():
    sentences = Sentence.find({ "type": "analyst" })
    texts = [sentence["text"] for sentence in sentences]
    pst = PorterStemmer()
    stemmed_texts = [[pst.stem(token) for token in word_tokenize(text.lower())] for text in texts]

    most_freq = most_common_keywords(stemmed_texts, 300)

    X_presences_common = words_presences(most_freq, stemmed_texts)
    # “complete”, “average”, “single”
    model = AgglomerativeClustering(distance_threshold=11.4, n_clusters=None, linkage="complete")

    print(np.any(X_presences_common, axis=1))
    model = model.fit(X_presences_common)
    print(model.labels_)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, p=3, get_leaves=True)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
    # stringBytes = io.BytesIO()
    # plt.savefig(stringBytes, format='png')
    plt.savefig("test.png", format='png')

# stringBytes.seek(0)
# base64Representation = base64.b64encode(stringBytes.read())
# encodedStr = str(base64Representation, "utf-8")
# print(encodedStr)
