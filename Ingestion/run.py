import argparse
import datetime
import logging
import os
import pytz
import shutil
import sys
import uuid

import numpy as np

from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Iterator


class IngestDocument:
    def __init__(
        self,
        embedding_model_name: str = "nomic-embed-text",
        embedding_batch_size: int = 100,
        chunking_strategy: str = "RecursiveCharacterTextSplitter",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        output_dir: str = "output",
    ):
        """
        Initialize document ingestion parameters.

        Args:
            embedding_model_name (str): Embedding model supported by Ollama to create the embeddings.
            chunking_strategy (str): Chunking Strategy to be used to generate the embeddings.
            chunk_size (int): The maximum size of each chunk.
            chunk_overlap (int): The overlap size between consecutive chunks.
            embedding_batch_size (int): Batch size for generating embeddings.
            output_dir (str): The output directory to save the embedding files.
        """

        self.embedding_model_name = embedding_model_name
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size

        self.output_dir = output_dir

        self.embedding_model = OllamaEmbeddings(model=self.embedding_model_name)

        unique_id = str(uuid.uuid4())
        time_stamp = datetime.datetime.now(pytz.timezone("US/Central"))
        time_stamp = time_stamp.strftime("%m_%d_%y_%H_%M_%S")
        self.identifier = "{}_{}".format(time_stamp, unique_id)

        if self.chunking_strategy == "RecursiveCharacterTextSplitter":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, self.output_dir)

        if os.path.exists(self.output_dir):
            print("Directory Path: {} exists, deleting it!".format(self.output_dir))
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir)
        print("Created directory: {}!".format(self.output_dir))

        self.log_filename = "{}/{}_ingestion.log".format(
            self.output_dir, self.identifier
        )
        self.logger = logging.getLogger("ingestion_logger")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        print("Log Filename: {}".format(self.log_filename))

    def read_document(self, file_path: str) -> str:
        """
        Reads a text document from the specified file path.

        Args:
            file_path (str): The path to the text document.

        Returns:
            text (str): The content of the text document as a string.
        """
        text = ""
        if file_path.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as text_file:
                    text = text_file.read()
                self.logger.info("Successfully read the document!")
                print("Successfully read the document!")
            except Exception as error:
                self.logger.error("Error while reading the document!")
                self.logger.error("Error: {}".format(error))
                print("Error while reading the document!")
                print("Error: {}".format(error))
        else:
            self.logger.error("Currently only text documents are supported!")
            print("Currently only text documents are supported!")
        return text

    def chunk_text(self, text):
        """
        Splits a given text into smaller chunks using a splitter.

        Args:
            text (str): The input text to split.

        Returns:
            chunked_text (list[str]): A list of text chunks.
        """
        chunked_text = self.text_splitter.split_text(text)
        return chunked_text

    def generate_batched_embeddings(
        self, chunked_text: List[str]
    ) -> Iterator[np.ndarray]:
        """
        Generate embeddings in batches.

        Args:
            chunked_text (List[str]): List of text chunks

        Yields:
            np.ndarray: Batch of embeddings
        """

        for idx in range(0, len(chunked_text), self.embedding_batch_size):
            batch_chunks = chunked_text[idx : idx + self.embedding_batch_size]
            batch_embeddings = []

            for chunk in batch_chunks:
                try:
                    embedding = self.embedding_model.embed_query(chunk)
                    batch_embeddings.append(embedding)
                except Exception as error:
                    self.logger.error("Error while embedding the chunk!")
                    self.logger.error("Error: {}".format(error))
                    print("Error while embedding the chunk!")
                    print("Error: {}".format(error))
            if batch_embeddings:
                yield np.array(batch_embeddings)

    def create_embeddings(self, chunked_text: List[str]) -> np.ndarray:
        """
        Consolidate embeddings from batched into a single numpy array.

        Args:
            chunked_text (List[str]): List of text chunks

        Returns:
            np.ndarray: Consolidated embeddings array
        """
        return np.concatenate(
            list(self.generate_batched_embeddings(chunked_text=chunked_text))
        )

    def ingest_document(self, file_path):
        """
        Main method to ingest and process a document.

        Args:
            file_path (str): Path to the document to be ingested
        """
        try:
            file_name = os.path.basename(file_path)
            file_name = file_name.split(".")[0]

            self.embedding_file_name = "{}/{}_{}.npy".format(
                self.output_dir, file_name, self.identifier
            )

            text = self.read_document(file_path=file_path)
            if not text:
                self.logger.error("No text to process!")
                print("No text to process!")
                sys.exit()

            chunked_text = self.chunk_text(text=text)
            self.logger.info(
                "Generated {} text chunks from the document!".format(len(chunked_text))
            )
            print(
                "Generated {} text chunks from the document!".format(len(chunked_text))
            )

            embeddings = self.create_embeddings(chunked_text=chunked_text)
            self.logger.info(
                "Generated {} embeddings from the chunks!".format(embeddings.shape)
            )
            print("Generated {} embeddings from the chunks!".format(embeddings.shape))

            try:
                np.save(self.embedding_file_name, embeddings, allow_pickle=True)
                self.logger.info(
                    "Saved the embeddings to {}!".format(self.embedding_file_name)
                )
                print("Saved the embeddings to {}!".format(self.embedding_file_name))
            except Exception as error:
                self.logger.error(
                    "Error saving the embeddings to {}!".format(
                        self.embedding_file_name
                    )
                )
                self.logger.error("Error: {}".format(error))
                print(
                    "Error saving the embeddings to {}!".format(
                        self.embedding_file_name
                    )
                )
                print("Error: {}".format(error))
        except Exception as error:
            self.logger.error("Error while ingesting the document!")
            self.logger.error("Error: {}".format(error))
            print("Error while ingesting the document!")
            print("Error: {}".format(error))


def get_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="Ingestion/input/report.txt",
        help="File path of your document",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="nomic-embed-text",
        help="Embedding model supported by Ollama to create the embeddings.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=100,
        help="Batch size to generate the embeddings.",
    )
    parser.add_argument(
        "--chunking_strategy",
        type=str,
        default="RecursiveCharacterTextSplitter",
        choices=["RecursiveCharacterTextSplitter"],
        help="Chunking Strategy to be used to generate the embeddings.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Maximum size of each chunk while splitting the document.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="The overlap size between consecutive chunks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory to save the embeddings.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("\n-------------------------------------")
    print("Ingesting the document!")
    args = get_args()
    print("Arguments: {}".format(vars(args)))

    ingestion = IngestDocument(
        embedding_model_name=args.embedding_model_name,
        embedding_batch_size=args.embedding_batch_size,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir,
    )

    ingestion.logger.info("-------------------------------------")
    ingestion.logger.info("Ingesting the document!")
    ingestion.logger.info("Arguments: {}".format(vars(args)))

    ingestion.ingest_document(file_path=args.file_path)

    ingestion.logger.info("Document has been ingested successfully!")
    ingestion.logger.info("-------------------------------------")
    print("Document has been ingested successfully!")
    print("-------------------------------------\n")
