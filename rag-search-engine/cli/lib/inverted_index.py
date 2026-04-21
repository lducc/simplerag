from csv import Error
from sqlite3 import NotSupportedError
from typing import List, Dict
from pathlib import Path
from collections import defaultdict
import os, pickle

from lib.utils import load_movies, DATA_PATH, CACHE_PATH
from lib.search_keyword import pre_process

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = CACHE_PATH/"index.pkl"
        self.docmap_path = CACHE_PATH/"docmap.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = pre_process(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, token: str) -> List[int]:
        return sorted(self.index.get(token, set()))

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

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise Error("Path does not exist")

        if not os.path.exists(self.docmap_path):
            raise Error("Path does not exist")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)



