import json
from pathlib import Path
import token
from typing import List, Dict
from lib.utils import pre_process
from lib.inverted_index import InvertedIndex

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

def matched_tokens_exists(movie_tokens: List[str], query_tokens: List[str]) -> bool:
    for q_token in query_tokens:
        for m_token in movie_tokens:
            if q_token in m_token:
                return True

    return False

def search(query: str, limit: int) -> List[str]:
    idx = InvertedIndex()
    idx.load()

    query_tokens = pre_process(query)
    seen, result = set(), []

    for token in query_tokens:
        movie_ids = idx.get_documents(token)
        for m_id in movie_ids:
            if m_id in seen:
                continue

            seen.add(m_id)
            doc = idx.docmap[m_id]
            result.append(doc)

            if len(result) >= limit:
                return result

    return result