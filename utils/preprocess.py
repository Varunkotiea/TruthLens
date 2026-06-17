import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def download_nltk_resources():
    for name in ["punkt", "stopwords"]:
        nltk.download(name, quiet=True)

download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
NEWS_NOISE = {
    "said", "says", "reuters", "ap", "cnn", "fox", "news", "report",
    "according", "via", "would", "also", "one", "two", "three",
    "new", "year", "time", "way", "day", "people", "could", "may",
}
ALL_STOP_WORDS = STOP_WORDS | NEWS_NOISE


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_filter(text: str, use_stemming: bool = True) -> list[str]:
    tokens = word_tokenize(text)
    return [
        STEMMER.stem(tok) if use_stemming else tok
        for tok in tokens
        if tok not in ALL_STOP_WORDS and len(tok) > 2
    ]


def preprocess(text: str, use_stemming: bool = True) -> str:
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned, use_stemming=use_stemming)
    return " ".join(tokens)


def preprocess_batch(texts: list[str], use_stemming: bool = True) -> list[str]:
    return [preprocess(t, use_stemming) for t in texts]