import pymongo
import os

myclient = pymongo.MongoClient("mongodb://root:password@{DB}:27017/".format(DB=os.getenv('DB', 'localhost')))
mydb = myclient["admin"]
Acquisition = mydb["Acquisitions"]
Document = mydb["Documents"]
Paragraph = mydb["Paragraphs"]
Keyword = mydb["Keywords"]
Model = mydb["Models"]
