import json
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/'data'

def load_movies() -> List[Dict]:
    with open(PATH/'movies.json', "r") as file:
        raw = json.load(file)

    return raw["movies"]

def load_stopwords() -> List[str]:
    with open(PATH/'stopwords.txt', "r") as file:
        raw = file.read().splitlines()

    return raw
