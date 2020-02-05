from flask import Flask, jsonify, request
import pickle
from api.db import *
import dateutil.parser
from bson.json_util import dumps
from bson import ObjectId
from flask import abort
from flask_cors import CORS
import GetOldTweets3 as got

app = Flask(__name__)
CORS(app)

def get_date(string_date):
    return dateutil.parser.isoparse(string_date)

@app.route('/predict', methods=['POST'])
def predict():
    model = Model.find_one({}, sort=[( '_id', pymongo.DESCENDING )])
    loaded_model = pickle.loads(model.model) # pickle.dumps(model) to get string
    sentences = request.get_json()
    X = [] # TODO: We have to decide if we use most freq words, words precences or words counts
    predicted = loaded_model.predict(X)
    result = zip(X, predicted)
    return jsonify(result)

@app.route('/train', methods=['POST'])
def train():
    # TODO: choose the model to train and the method
    # clf = svm.SVC()
    # clf.fit(X, y)
    # model = pickle.dumps(clf)
    # Model.insert_one({ "model": model })
    return "model trained";

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
        sentence = request.get_json()
        if sentence["type"] == 'twitter':
            sentence = {
                "text": sentence["text"],
                "class": sentence["class"],
                "type": sentence["type"]
            }
        else:
            sentence = {
                "text": sentence["text"],
                "class": sentence["class"],
                "type": sentence["type"],
                "document_id": ObjectId(sentence["document_id"])
            }
            pipeline = [
                { '$unwind': '$documents' },
                { '$match': { 'documents._id': sentence["document_id"] } }
            ]
            document = list(Acquisition.aggregate(pipeline))
            if len(document) != 1:
                abort(400) # Document not foudn
        return {"_id": str(Sentence.insert_one(sentence).inserted_id) }

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
        return str(Keyword.insert_many(keyword).inserted_ids)

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
