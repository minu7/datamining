import xlrd
import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import functools
import pandas as pd
import re

if len(sys.argv) != 2:
    print('you must provide the output file')
    exit()

data = xlrd.open_workbook(filename = 'app/data/initialdata.xlsx')
ceo_sheet = data.sheets()[1] # the seconds sheets contains the data about the analyst
# we are interested in the text o
ceo = [[row[3].value, row[4].value] for row in ceo_sheet.get_rows()]
# column in order: keywords, text, sentiment
# first row is header
ceo = ceo[1 : len(ceo)]

# tokenize and lemmatize the sentences and get a plain array with all words
lemmatizer = WordNetLemmatizer()
ceo = [[[lemmatizer.lemmatize(x) for x in word_tokenize(row[0].lower())], row[1]] for row in ceo]

words = functools.reduce(lambda x, y: x+y, [x[0] for x in ceo], [])

# remove stop word for meaningful word frequencies
a = stopwords.words('english')
a = a + [',', '.', '$', '\'s', '%', '&', 'n\'t', '--', '’', 'would', 'also', '...', ';', '\'ll', 'u', 'ha', '\'ve', '\'re'];
words = [x for x in words if x not in a]

# count frequencies and get N most common words
keywords = [re.split(',|;', row[2].value.lower()) for row in ceo_sheet.get_rows()]
keywords = keywords[1 : len(keywords)]
keywords = functools.reduce(lambda x, y: x+[y], [x[0] for x in keywords], [])
keywords = [lemmatizer.lemmatize(x) for x in keywords]
keywords = list(set(keywords)) # to avoid keywords repetition

csv = []
csv.append([] + keywords + ['class'])
for row in ceo:
    print(row[0])
    csv.append([1 if keyword in row[0] else 0 for keyword in keywords] + [row[1]])

pd.DataFrame(csv).to_csv(sys.argv[1], header=None, index=None)
