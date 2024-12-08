import argparse

from utils import *
from llm_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.2:3b",
        help="LLM supported by Ollama to power the Chatbot",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="nomic-embed-text",
        help="Embedding model supported by Ollama to power the Chatbot",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=0.1,
        help="Temperature value for the LLM, closer to 0 more realistic",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Context window for the LLM"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="csv",
        choices=["csv", "xml"],
        help="File type of your document to to perform RAG",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/combined_adult_protocols.csv",
        help="File path of your document",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Size of each chunk while splitting the document",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="Overlap limit while performing the splitting",
    )
    parser.add_argument(
        "--search_type",
        type=str,
        default="similarity",
        choices=["similarity"],
        help="Method to perform the search",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Top n results of the search will be returned",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(
        "\n-------------------------------------\nRunning RAG powered Chatbot!\n-------------------------------------\n"
    )
    args = get_args()
    print("Arguments: {}".format(vars(args)))
    print("-------------------------------------")
    logger.info("Running RAG powered Chatbot!")
    logger.info("Arguments: {}".format(vars(args)))
    logger.info("-------------------------------------")
    run_chatbot(args=args, logger=logger)
    logger.info("Chatbot execution completed!")
    print(
        "\n-------------------------------------\nChatbot execution completed!\n-------------------------------------\n"
    )
