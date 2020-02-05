import sys
import os
from os.path import dirname, abspath

sys.path.insert(1, dirname(dirname(abspath(__file__)))) # this is needed for import Acquisition (we need to add the root folder in module search paths)

import xlrd
from api.db import Acquisition
from datetime import datetime
from excel_date_to_datetime import excel_to_datetime
import re

def import_documents():
    data = xlrd.open_workbook(filename = 'app/data/initialdata.xlsx')
    documents = data.sheets()[3]
    rows = documents.get_rows()

    for i, val in enumerate(rows):
        if i == 0:
            continue # discard header

        link = val[0].value
        date = re.split(',|;', val[1].value.lower())[0].strip()
        date = datetime.strptime(date, '%m/%d/%y')
        source = re.split(',|;', val[1].value.lower())[1].strip().replace("\"", "")
        type = val[2].value.lower().strip()
        op_id = int(val[3].value)

        document = {
            "link": link,
            "date": date,
            "source": source,
            "type": type,
            "_id": ObjectId()
        }
        Acquisition.update_one({ "op_id": op_id }, { '$push': {'documents': document} })
