from csv import Error
from multiprocessing import Value
from socket import TCP_DEFER_ACCEPT
from sqlite3 import NotSupportedError
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import os, pickle, math

from lib.utils import load_movies, DATA_PATH, CACHE_PATH
from lib.utils import pre_process
from lib.utils import BM25_B, BM25_K

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_freqs: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

        self.index_path = CACHE_PATH/"index.pkl"
        self.docmap_path = CACHE_PATH/"docmap.pkl"
        self.freq_path = CACHE_PATH/"term_frequencies.pkl"
        self.doc_length_path = CACHE_PATH/"doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = pre_process(text)
        counter = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_freqs[doc_id] = counter
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        total = sum(self.doc_lengths.values())
        amount = len(self.doc_lengths)

        if amount == 0:
            return 0.0

        return total / amount

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
        doc_count = len(self.docmap) #N
        term_doc_count = len(self.index[token]) #df
        #log(N / df) (with smoothing)
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_tf(self, doc_id: int, term: str, K: float = BM25_K, B: float = BM25_B) -> float:
        """
        K, B: BM25 params for controling term freq scaling
        Ex (K): A document that has 200 "game" terms should not mean that it is
        2x as more important for retrieval than the one that has 100 "game" terms
        -> K is for controlling the saturation effect beyond a certain point

        Ex (B): A document that is longer contains more words -> Boost their TF scores
        without being more relevant
        -> B (0, 1) is a tunable param that controls how much we care about doc length. """
        if doc_id not in self.term_freqs:
            return 0

        tokens = pre_process(term)
        if len(tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token")

        # Implementation without document normalization param B
        # tf = self.get_tf(doc_id, term)
        # bm25_tf = (tf * (K + 1) / (tf + K))

        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)

        # print(self.term_freqs[doc_id], sum(self.term_freqs[doc_id].values()), doc_length, avg_doc_length)
        length_norm = 1 - B + B * (doc_length / avg_doc_length)

        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (K + 1) / (tf + K * length_norm))

        return bm25_tf

    def get_bm25_idf(self, term: str) -> float:
        tokens = pre_process(term)
        if len(tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token")

        token = tokens[0]
        doc_count = len(self.docmap) #n
        term_doc_count = len(self.index[token]) #df
        #log((n - df + 0.5) / (df + 0.5) + 1)

        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf_value = self.get_tf(doc_id, term)
        idf_value = self.get_idf(term)

        return tf_value * idf_value

    def get_bm25_score(self, doc_id: int, term: str) -> float:
        bm25_tf_value = self.get_bm25_tf(doc_id, term)
        bm25_idf_value = self.get_bm25_idf(term)

        return bm25_idf_value * bm25_tf_value

    def bm25_search(self, query: str, limit: int = 5) -> List[Tuple[int, float]]:
        tokens = pre_process(query)
        bm25_scores = defaultdict(float)

        for token in tokens:
            for doc_id in self.docmap:
                bm25_scores[doc_id] += self.get_bm25_score(doc_id, token)

        scores = sorted(bm25_scores.items(), key=lambda x: -x[1])

        return scores[:limit]

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

        with open(self.doc_length_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index file does not exist")

        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError("Docmap file does not exist")

        if not os.path.exists(self.freq_path):
            raise FileNotFoundError("Frequency file does not exist")

        if not os.path.exists(self.doc_length_path):
            raise FileNotFoundError("Doc lengths file does not exist")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.freq_path, "rb") as f:
            self.term_freqs = pickle.load(f)

        with open(self.doc_length_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

