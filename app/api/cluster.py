import numpy as np

from db import Sentence
from db import Keyword

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from utility import transformer_embeddings
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
    pipeline = [
        { '$group': { '_id': '$document_id', 'sentences': { '$push': '$text' } } },
        { '$project': { '_id': 1, 'text': { '$reduce': { 'input': '$sentences', 'initialValue': '', 'in': { '$concat': ['$$value', ' ', '$$this'] } } } } }
    ]
    document = list(Sentence.aggregate(pipeline))
    texts = [sentence["text"] for sentence in document]

    X_presences_common = transformer_embeddings(texts)
    # “complete”, “average”, “single”
    model = AgglomerativeClustering(distance_threshold=11.4, n_clusters=None, linkage="average", affinity="cosine")

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
    plt.savefig("test.png", format='png', dpi=300)

# stringBytes.seek(0)
# base64Representation = base64.b64encode(stringBytes.read())
# encodedStr = str(base64Representation, "utf-8")
# print(encodedStr)

cluster()
