import sys
import os
from os.path import dirname, abspath

sys.path.insert(1, dirname(dirname(abspath(__file__)))) # this is needed for import Acquisition (we need to add the root folder in module search paths)

import xlrd
from api.db import Acquisition
from datetime import datetime
from excel_date_to_datetime import excel_to_datetime
import pymongo
# from bson.objectid import ObjectId
# to run this: docker-compose run flask python app/excel_to_mongo/acquisitions.py

data = xlrd.open_workbook(filename = 'app/data/initialdata.xlsx')
acquisitions = data.sheets()[0]

rows = acquisitions.get_rows()

for i, val in enumerate(rows):
    if i == 0:
        continue # discard header

    document = {
        "annuncement_date" : excel_to_datetime(val[0].value),
        "signing_date": excel_to_datetime(val[1].value),
        "status": val[2].value.lower().strip(),
        "acquiror": {
            "name": val[4].value,
            "ticker": val[5].value.lower().strip(),
            "state": val[6].value.lower().strip(),
        },
        "target": {
            "name": val[7].value,
            "ticker": val[8].value.lower().strip(),
            "state": val[9].value.lower().strip()
        },
        "op_id": val[10].value,
        "documents": []
    }
    Acquisition.insert_one(document)
# ad unique index on op_id
Acquisition.create_index(
    [("op_id", pymongo.ASCENDING)],
    unique=True
)
