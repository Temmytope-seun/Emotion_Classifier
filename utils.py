# utils.py

import re
import string
import nltk
import contractions
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Remove text in square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove newlines
    text = text.replace('\n', ' ').replace('\r', '')

    # Remove words with numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove apostrophes
    text = text.replace("'", "")

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back to string
    return ' '.join(filtered_tokens)
