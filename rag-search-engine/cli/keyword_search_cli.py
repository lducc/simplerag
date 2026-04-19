import argparse
from lib.search_keyword import search
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Seach CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies (BM25)")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            query_results = search(query=query, top_k=5)

            for i, result in enumerate(query_results):
                print(f"{i}. {result}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()