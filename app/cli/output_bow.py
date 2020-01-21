import xlrd
import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import functools
import pandas as pd

if len(sys.argv) != 3 or int(sys.argv[1]) <= 0:
    print('you must provide number of words from most common to select')
    exit()

data = xlrd.open_workbook(filename = 'app/data/initialdata.xlsx')
ceo = data.sheets()[1] # the seconds sheets contains the data about the analyst
# we are interested in the text o
ceo = [[row[3].value, row[4].value] for row in ceo.get_rows()]
# column in order: text, sentiment
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
fdist = FreqDist(words)
most_freq = fdist.most_common(int(sys.argv[1]))

most_freq = [x[0] for x in most_freq]
csv = []
csv.append([] + most_freq + ['class'])
for row in ceo:
    csv.append([1 if frequent_word in row[0] else 0 for frequent_word in most_freq] + [row[1]])
print(ceo)
pd.DataFrame(csv).to_csv(sys.argv[2], header=None, index=None)
