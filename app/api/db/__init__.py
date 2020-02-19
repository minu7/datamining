import pymongo
import os

myclient = pymongo.MongoClient("mongodb://root:password@{DB}:27017/".format(DB=os.getenv('DB', 'localhost')))
mydb = myclient["admin"]
Acquisition = mydb["Acquisitions"]
Sentence = mydb["Sentences"]
Keyword = mydb["Keywords"]
Model = mydb["Models"]
Attribute = mydb["Attributes"]
