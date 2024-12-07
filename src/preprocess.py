import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Preprocesses the given text by removing special characters, 
    lowercasing, removing stop words, and stemming.

    Args:
        text: The text to preprocess.

    Returns:
        The preprocessed text.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Lowercase
    words = [word for word in text.split() if word not in stop_words]  # Remove stop words
    stemmed_words = [stemmer.stem(word) for word in words]  # Stemming
    return " ".join(stemmed_words)
