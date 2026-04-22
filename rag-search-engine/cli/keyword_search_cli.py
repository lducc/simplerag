import argparse
from ast import In
from re import sub
from lib.search_keyword import search
from lib.inverted_index import InvertedIndex
import math
from lib.utils import BM25_B, BM25_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Seach CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Compute TF value")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    search_parser = subparsers.add_parser("search", help="Search movies (BM25)")
    search_parser.add_argument("query", type=str, help="Search query")

    idf_parser = subparsers.add_parser("idf", help="Compute IDF value")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Compue TF-IDF value")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Compue BM25 TF value")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser.add_argument("k", type=float, nargs='?', default=BM25_K, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Compute BM25 IDF score")
    bm25_idf_parser.add_argument("term", type=str, help="Term")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "search":
            index = InvertedIndex()
            try:
                index.load()
            except:
                print("File does not exist")
            query = args.query
            print(f"Searching for: {query}")
            query_results = search(query=query, limit=5)

            for i, result in enumerate(query_results):
                print(f"{i}. {result}")
        case "tf":
            index = InvertedIndex()
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case "idf":
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tf_idf = index.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25_tf = index.get_bm25_tf(args.doc_id, args.term, args.k, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case "bm25idf":
            index = InvertedIndex()
            index.load()
            bm25_idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25search":
            index = InvertedIndex()
            index.load()
            query_results = index.bm25_search(args.query, args.limit)

            for i, (doc_id, score) in enumerate(query_results):
                print(f"{i}. ({doc_id}) {index.docmap[doc_id]["title"]} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()