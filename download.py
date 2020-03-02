import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

model = SentenceTransformer('bert-base-nli-mean-tokens')
SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
