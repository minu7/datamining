from flask import Flask, jsonify, request
import pickle
from db import *

app = Flask(__name__)

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
    return "";

@app.route('/acquisition/<acquisition_id>', methods=['POST', 'GET'])
def acquisition(acquisition_id):
    if request.method == 'POST':
        acquisition = request.get_json()
        document = {
            "annuncement_date" : acquisition["annuncement_date"],
            "signing_date": acquisition["signing_date"],
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
        return jsonify(Acquisition.insert_one(document))
    if acquisition_id is not None:
        acquisition = Acquisition.find_one({ '_id': acquisition_id })
        return jsonify(acquisition)
    acquisition = Acquisition.find()
    return jsonify(acquisition)


@app.route('/document/<acquisition_id>', methods=['POST'])
    def document(acquisition_id):
        documents = request.get_json()
        d = {
            "link": documents["link"],
            "date": documents["date"],
            "source": documents["source"],
            "type": documents["type"]
        }
        return Acquisition.update_one({ "_id": acquisition_id }, { '$push': {'documents': d} })

@app.route('/sentence/<sentence_id>', methods=['POST', 'GET'])
def sentence(sentence_id):
    if request.method == 'POST':
        sentence = request.get_json()
        sentence = {
            "text": sentence["text"],
            "class": sentence["class"],
            "type": sentence["type"],
            "document_id": sentence["document_id"]
        }
        pipeline = [
            { '$unwind': '$documents' },
            { '$match': { 'documents._id': sentence["document_id"] } }
        ]
        document = Acquisition.aggregate(pipeline)
        if len(document) != 1:
            abort(400) # Document not foudn
        return Sentence.insert_one(sentence)

    document_id = request.args.get('document_id')
    if sentence_id is not None:
        return jsonify(Sentence.find_one({ '_id': sentence_id }))

    if document_id is not None:
        return jsonify(Sentence.find({ 'document_id': document_id }))

    return jsonify(Sentence.find({}))

@app.route('/keyword', methods=['POST', 'GET'])
def keyword():
    if request.method == 'POST':
        sentence = request.get_json()
        keyword = {
            "value": keyword,
            "type": "analyst"
        }
        return jsonify(Keyword.insert_one(document))

    type = request.args.get('type')
    if type is not None:
        return jsonify(Keyword.find({ 'type': type }))
    return jsonify(Keyword.find({ }))

# TODO: testing
