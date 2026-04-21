import json
from pathlib import Path
from typing import List, Dict
from nltk.stem import PorterStemmer
import string

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT/'data'
CACHE_PATH = ROOT/'cache'

def load_movies() -> List[Dict]:
    with open(DATA_PATH/'movies.json', "r") as file:
        raw = json.load(file)

    return raw["movies"]

def load_stopwords() -> List[str]:
    with open(DATA_PATH/'stopwords.txt', "r") as file:
        raw = file.read().splitlines()

    return raw


def pre_process(text: str) -> List[str]:
    """
        Text -> Lowercase -> Remove puncs
        -> Tokenize -> Remove stop words
        -> Stemming
    """
    if not text: return []

    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()

    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]

    return tokens