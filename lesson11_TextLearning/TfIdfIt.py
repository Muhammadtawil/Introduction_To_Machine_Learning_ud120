

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
vectorizer.fit_transform(word_data)

feature_names = vectorizer.get_feature_names()

print('Number of different words: {0}'.format(len(feature_names)))