from flask import Flask, jsonify, request
from api.db import *
import dateutil.parser
from bson.json_util import dumps
from bson import ObjectId
from flask import abort
from flask_cors import CORS
import GetOldTweets3 as got
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from nltk.tokenize import word_tokenize
from api.utility import words_presences
from sklearn.metrics import confusion_matrix
from joblib import dump, load

from sentence_transformers import SentenceTransformer
from api.utility import plot_dendrogram
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import io
import base64

app = Flask(__name__)
CORS(app)

transformer = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

def get_date(string_date):
    return dateutil.parser.isoparse(string_date)

@app.route('/predict', methods=['POST'])
def predict():
    params = request.get_json()
    model = Model.find_one({ "type": params["model"]  }, sort=[( '_id', pymongo.DESCENDING )])
    clf = load('models/' + str(model["_id"]))
    sentences = params["sentences"]
    X = transformer.encode(sentences)
    predicted = clf.predict(X)
    result = list(zip(sentences, predicted))
    result = [{ "sentence": x[0], "prediction": int(x[1]) } for x in result]
    return jsonify(result)

@app.route('/train', methods=['POST'])
def train():
    sentences = Sentence.find({ "type": "analyst" })
    sentences = [sentence for sentence in sentences]
    X_presences_common = transformer.encode([sentence["text"] for sentence in sentences])
    y = [sentence["class"] for sentence in sentences]

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X_presences_common, y, cv=10)
    y_pred = cross_val_predict(clf, X_presences_common, y, cv=10)
    clf = clf.fit(X_presences_common, y)
    conf_mat = confusion_matrix(y, y_pred)
    svmDocument = { "type": "svm", "accuracy": scores.mean(), "std": scores.std(), "confusion_matrix": conf_mat.tolist() }
    dump(clf, 'models/' + str(Model.insert_one(svmDocument).inserted_id))
    del svmDocument["_id"]

    clf = knn(n_neighbors=3, metric='minkowski')
    scores = cross_val_score(clf, X_presences_common, y, cv=10)
    y_pred = cross_val_predict(clf, X_presences_common, y, cv=10)
    clf = clf.fit(X_presences_common, y)
    conf_mat = confusion_matrix(y, y_pred)
    knnDocument = { "type": "knn", "accuracy": scores.mean(), "std": scores.std(), "confusion_matrix": conf_mat.tolist() }
    dump(clf, 'models/' + str(Model.insert_one(knnDocument).inserted_id))
    del knnDocument["_id"]

    clf = lr()
    scores = cross_val_score(clf, X_presences_common, y, cv=10)
    y_pred = cross_val_predict(clf, X_presences_common, y, cv=10)
    clf = clf.fit(X_presences_common, y)
    conf_mat = confusion_matrix(y, y_pred)
    logisticDocument = { "type": "logistic", "accuracy": scores.mean(), "std": scores.std(), "confusion_matrix": conf_mat.tolist() }
    dump(clf, 'models/' + str(Model.insert_one(logisticDocument).inserted_id))
    del logisticDocument["_id"]

    return jsonify({ "svm": svmDocument, "knn": knnDocument, "logistic": logisticDocument })

@app.route('/cluster', methods=['POST'])
def cluster():
    threshold = float(request.get_json()["threshold"])
    affinity = request.get_json()["affinity"]
    linkage = request.get_json()["linkage"]
    pipeline = [
        { '$group': { '_id': '$document_id', 'sentences': { '$push': '$text' } } },
        { '$project': { '_id': 1, 'text': { '$reduce': { 'input': '$sentences', 'initialValue': '', 'in': { '$concat': ['$$value', ' ', '$$this'] } } } } }
    ]
    document = list(Sentence.aggregate(pipeline))
    texts = [sentence["text"] for sentence in document]

    X_presences_common = transformer.encode(texts)
    model = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, linkage=linkage, affinity=affinity)
    model = model.fit(X_presences_common)
    result = list(zip(texts, model.labels_))
    result = [{ "text": x[0], "cluster": int(x[1]) } for x in result]
    plt.title("Hierarchical Clustering {} {} dendrogram".format(affinity, linkage))
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, p=3, get_leaves=True)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    stringBytes = io.BytesIO()
    plt.savefig(stringBytes, format='png', dpi=300)
    base64Representation = base64.b64encode(stringBytes.getvalue())
    encodedStr = base64Representation.decode()
    plt.clf()
    plt.cla()
    plt.close()
    return dumps({ "result": result, "image": encodedStr })


@app.route('/acquisition', methods=['POST', 'GET'], defaults={'acquisition_id': None})
@app.route('/acquisition/<acquisition_id>', methods=['POST', 'GET'])
def acquisition(acquisition_id):
    if request.method == 'POST':
        acquisition = request.get_json()
        document = {
            "annuncement_date" : get_date(acquisition["annuncement_date"]),
            "signing_date": get_date(acquisition["signing_date"]),
            "status": acquisition["status"].lower().strip(),
            "acquiror": {
                "name": acquisition["acquiror"]["name"],
                "ticker": acquisition["acquiror"]["ticker"].lower().strip(),
                "state": acquisition["acquiror"]["state"].lower().strip(),
            },
            "target": {
                "name": acquisition["target"]["name"],
                "ticker": acquisition["target"]["ticker"].lower().strip(),
                "state": acquisition["target"]["state"].lower().strip()
            },
            "documents": []
        }
        return {"_id": str(Acquisition.insert_one(document).inserted_id) }
    if acquisition_id is not None:
        acquisition = Acquisition.find_one({ '_id': ObjectId(acquisition_id) })
        return dumps(acquisition)
    acquisition = Acquisition.find()
    return dumps(acquisition)


@app.route('/document', methods=['POST'])
def document():
    documents = request.get_json()
    acquisition_id = documents["acquisition_id"]
    d = {
        "title": documents["title"],
        "link": documents["link"],
        "date": get_date(documents["date"]),
        "source": documents["source"],
        "type": documents["type"],
        "_id": ObjectId()
    }
    return { 'updated': Acquisition.update_one({ "_id": ObjectId(acquisition_id) }, { '$push': {'documents': d} }).modified_count > 0 }

@app.route('/sentence', methods=['POST', 'GET'], defaults={'sentence_id': None})
@app.route('/sentence/<sentence_id>', methods=['POST', 'GET'])
def sentence(sentence_id):
    if request.method == 'POST':
        sentences = request.get_json()
        if sentences["type"] == 'twitter':
            sentences = [{ "text": sentence["text"], "class": sentence["class"], "type": sentences["type"]} for sentence in sentences["sentences"]]
        else:
            sentences = [{"text": sentence["text"], "class": sentence["class"], "type": sentences["type"], "document_id": ObjectId(sentences["document_id"])} for sentence in sentences["sentences"]]
            pipeline = [
                { '$unwind': '$documents' },
                { '$match': { 'documents._id': sentences["document_id"] } }
            ]
            document = list(Acquisition.aggregate(pipeline))
            if len(document) != 1:
                abort(400) # Document not found
        Sentence.insert_many(sentences)
        return dumps({ 'inserted': True })

    document_id = request.args.get('document_id')
    if sentence_id is not None:
        return dumps(Sentence.find_one({ '_id': ObjectId(sentence_id) }))

    if document_id is not None:
        return dumps(Sentence.find({ 'document_id': ObjectId(document_id) }))

    return dumps(Sentence.find({}))

@app.route('/keyword', methods=['POST', 'GET'])
def keyword():
    if request.method == 'POST':
        keyword = request.get_json()
        Keyword.delete_many({})
        return dumps(Keyword.insert_many(keyword).inserted_ids)

    type = request.args.get('type')
    if type is not None:
        return dumps(Keyword.find({ 'type': type }))
    return dumps(Keyword.find({ }))

@app.route('/tweets/<username>', methods=['GET'])
def tweets(username):
    tweetCriteria = None
    if request.args.get('since') is not None and request.args.get('until') is not None:
        tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(50).setSince(request.args.get('since')).setUntil(request.args.get('until'))
    elif request.args.get('since') is not None:
        tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(50).setSince(request.args.get('since'))
    else:
        tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(50)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweets = [tweet.text for tweet in tweets]
    return jsonify({"tweets": tweets})
