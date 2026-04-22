from csv import Error
from sqlite3 import NotSupportedError
from typing import List, Dict
from pathlib import Path
from collections import defaultdict, Counter
import os, pickle, math

from lib.utils import load_movies, DATA_PATH, CACHE_PATH
from lib.utils import pre_process

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_freqs: dict[int, Counter] = {}
        self.index_path = CACHE_PATH/"index.pkl"
        self.docmap_path = CACHE_PATH/"docmap.pkl"
        self.freq_path = CACHE_PATH/"term_frequencies.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = pre_process(text)
        counter = Counter(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

        self.term_freqs[doc_id] = counter

    def get_documents(self, token: str) -> List[int]:
        return sorted(self.index.get(token, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_freqs:
            return 0

        token = pre_process(term)
        if len(token) != 1:
            raise ValueError("Term must tokenize to exactly one token")

        return self.term_freqs[doc_id].get(token[0], 0)

    def get_idf(self, term: str) -> float:
        tokens = pre_process(term)
        if len(tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            movie_id = movie["id"]
            movie_title = movie["title"]
            movie_desc = movie["description"]

            self.docmap[movie_id] = movie
            self.__add_document(movie_id, f'{movie_title} {movie_desc}')

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.freq_path, "wb") as f:
            pickle.dump(self.term_freqs, f)

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index file does not exist")

        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError("Docmap file does not exist")

        if not os.path.exists(self.freq_path):
            raise FileNotFoundError("Frequency file does not exist")


        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.freq_path, "rb") as f:
            self.term_freqs = pickle.load(f)


