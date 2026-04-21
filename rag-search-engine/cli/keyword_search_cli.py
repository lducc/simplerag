import argparse
from lib.search_keyword import search
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Seach CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")
    search_parser = subparsers.add_parser("search", help="Search movies (BM25)")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            docs = index.get_documents("merida")
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()