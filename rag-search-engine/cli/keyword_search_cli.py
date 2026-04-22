import argparse
from lib.search_keyword import search
from lib.inverted_index import InvertedIndex
import math

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()