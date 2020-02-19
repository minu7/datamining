import sys
import os
from os.path import dirname, abspath

sys.path.insert(1, dirname(dirname(abspath(__file__)))) # this is needed for import Acquisition (we need to add the root folder in module search paths)

import xlrd
from api.db import Sentence
from api.db import Acquisition
from datetime import datetime
from excel_date_to_datetime import excel_to_datetime
import re


def import_sentences():
    data = xlrd.open_workbook(filename = 'app/data/initialdata.xlsx')
    documents = data.sheets()[1]
    rows = documents.get_rows()

    for i, val in enumerate(rows):
        if i == 0:
            continue # discard header

        date = re.split(',|;', val[1].value.lower())[0].strip()
        date = datetime.strptime(date, '%m/%d/%y')
        source = re.split(',|;', val[1].value.lower())[1].strip().replace("\"", "")
        text = val[3].value # text saved as is, we perform preprocessing during training for now,
        # we have to decide what is better: preprocessing now or before training?
        class_ = int(val[4].value)

        d = {
            "text": text,
            "class": class_,
            "type": "ceo",
        }
        pipeline = [
            { '$unwind': '$documents' },
            { '$match': { 'documents.date': date, 'documents.source': source } }
        ]
        document = Acquisition.aggregate(pipeline)
        document = list(document)
        if len(document) != 1:
            raise Exception("duplicate error")
            exit()
        d["document_id"] = document[0]["documents"]["_id"]
        Sentence.insert_one(d)


    documents = data.sheets()[2]
    rows = documents.get_rows()

    for i, val in enumerate(rows):
        if i == 0:
            continue # discard header

        date = re.split(',|;', val[1].value.lower())[0].strip()
        date = datetime.strptime(date, '%m/%d/%y')
        source = re.split(',|;', val[1].value.lower())[1].strip().replace("\"", "")
        text = val[3].value # text saved as is, we perform preprocessing during training for now,
        # we have to decide what is better: preprocessing now or before training?
        class_ = int(val[4].value)

        d = {
            "text": text,
            "class": class_,
            "type": "analyst",
        }
        pipeline = [
            { '$unwind': '$documents' },
            { '$match': { 'documents.date': date, 'documents.source': source } }
        ]
        document = Acquisition.aggregate(pipeline)
        document = list(document)
        if len(document) != 1:
            print(document)
            print(d)
            raise Exception("duplicate error")
        else:
            d["document_id"] = document[0]["documents"]["_id"]
        Sentence.insert_one(d)
