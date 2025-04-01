from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_stopwords(text):
    """
    remove stop words using nltk but currently not functional yet
    """
    # nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    _tokens = word_tokenize(text)
    _filtered = [w for w in _tokens if w not in stop_words]
    return _filtered
