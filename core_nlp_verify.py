import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text = "Natural Language Processing (NLP) is a fascinating field of study! It involves analyzing and understanding human language."
filtered_tokens = [token.lower() for token in word_tokenize(text) if token.lower() not in stop_words and token.isalpha()]
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]

corpus = [
    "I love NLP.",
    "NLP is amazing.",
    "I enjoy learning new things in NLP."
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_bigram = bigram_vectorizer.fit_transform(corpus)

def text_preprocessing_pipeline(txt):
    clean_text = re.sub(r"[^a-zA-Z\s]", " ", txt.lower())
    tokens = word_tokenize(clean_text)
    filtered = [token for token in tokens if token not in stop_words and token.isalpha()]
    return [lemmatizer.lemmatize(token, pos='v') for token in filtered]

processed_text = text_preprocessing_pipeline(text)

sentence = "The cats are playing with the mice in the garden."
filtered_tokens_2 = [token.lower() for token in word_tokenize(sentence) if token.lower() not in stop_words and token.isalpha()]
stemmed_tokens_2 = [stemmer.stem(token) for token in filtered_tokens_2]
lemmatized_tokens_2 = [lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens_2]

print('CORE_NLP_OK')
print('filtered_tokens:', filtered_tokens)
print('stemmed_tokens:', stemmed_tokens)
print('lemmatized_tokens:', lemmatized_tokens)
print('X shape:', X.shape)
print('X_tfidf shape:', X_tfidf.shape)
print('X_bigram shape:', X_bigram.shape)
print('processed_text:', processed_text)
print('sentence comparison:', filtered_tokens_2, stemmed_tokens_2, lemmatized_tokens_2)
