import json
from pathlib import Path
import token
from typing import List, Dict
from lib.search_utils import load_movies, load_stopwords
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string


def pre_process(text: str) -> List[str]:
    """
        Text -> Lowercase -> Remove puncs
        -> Tokenize -> Remove stop words
        -> Stemming
    """
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()

    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]

    return tokens

def matched_tokens_exists(movie_tokens: List[str], query_tokens: List[str]) -> bool:
    for q_token in query_tokens:
        for m_token in movie_tokens:
            if q_token in m_token:
                return True

    return False

def search(query: str, top_k: int) -> List[str]:
    movies = load_movies()
    search_result = []
    query_tokens = pre_process(query)

    for movie in movies:
        movie_title = movie.get('title')
        movie_tokens = pre_process(movie_title)
        if matched_tokens_exists(movie_tokens, query_tokens):
            search_result.append(movie_title)

        if len(search_result) > top_k:
            break

    return search_result