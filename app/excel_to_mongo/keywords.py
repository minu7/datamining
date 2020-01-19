import sys
import os
from os.path import dirname, abspath

sys.path.insert(1, dirname(dirname(abspath(__file__)))) # this is needed for import Acquisition (we need to add the root folder in module search paths)

import xlrd
from api.db import Keyword
from datetime import datetime
from excel_date_to_datetime import excel_to_datetime
import re

def import_keywords():
    data = xlrd.open_workbook(filename = 'app/data/initialdata.xlsx')
    documents = data.sheets()[1]
    rows = documents.get_rows()
    keywords = []
    # TODO: see if better to stem or to lemm now or later (maybe now)
    for i, row in enumerate(rows):
        if i == 0:
            continue # discard header

        keywords = keywords + re.split(',|;', row[2].value.lower())

    documents = data.sheets()[2]
    rows = documents.get_rows()

    for i, row in enumerate(rows):
        if i == 0:
            continue # discard header

        keywords = keywords + re.split(',|;', row[2].value.lower())

    keywords = list(set(keywords)) # to delete duplicated keywords

    for keyword in keywords:
        Keyword.insert_one({
            "value": keyword
        })
