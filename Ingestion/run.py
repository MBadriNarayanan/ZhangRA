import argparse
import datetime

# import flywheel
import gc
import logging
import os
import pytz
import shutil
import sys
import torch
import uuid

import numpy as np
from transformers import logging as tf_logging
from typing import List, Iterator

from torch.amp import autocast
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel

from langchain.text_splitter import RecursiveCharacterTextSplitter

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class IngestDocument:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy: str = "RecursiveCharacterTextSplitter",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_batch_size: int = 100,
        output_dir: str = "ingestion_output",
        device: str = None,
        max_length: int = 1024,
        mixed_precision: bool = True,
    ):
        """
        Initialize document ingestion parameters.

        Args:
            embedding_model_name (str): Embedding model supported by HuggingFace to create the embeddings.
            chunking_strategy (str): Chunking Strategy to be used to generate the embeddings.
            chunk_size (int): The maximum size of each chunk.
            chunk_overlap (int): The overlap size between consecutive chunks.
            embedding_batch_size (int): Batch size for generating embeddings.
            output_dir (str): The output directory to save the embedding files.
            device (str): Device to run the model on ('cuda', 'mps', 'cpu').
            max_length (int): Maximum sequence length for the models.
            mixed_precision (bool): Boolean to indicate whether to use mixed precision inference.
        """

        self.embedding_model_name = embedding_model_name
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = embedding_batch_size
        self.max_length = max_length
        self.mixed_precision = mixed_precision and torch.cuda.is_available()

        self.assign_device(device=device)
        self.assign_chunking_strategy()
        self.initialize_model()
        self.create_identifier()
        self.create_output_dir(output_dir=output_dir)
        self.setup_logging()

        print(
            "Initialized model: {} on device: {}!".format(
                self.embedding_model_name, self.device
            )
        )

        if self.mixed_precision:
            self.logger.info("Using mixed precision!")
            print("Using mixed precision!")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(
                "Using GPU: {} with {:.2f} GB memory!".format(gpu_name, gpu_memory)
            )
            print("Using GPU: {} with {:.2f} GB memory!".format(gpu_name, gpu_memory))

    def assign_device(self, device: str):
        """
        Assign a device for computation, based on user input or system availability.

        This method selects the computation device (e.g., CUDA, MPS, or CPU) to be used
        by the model. If no device is explicitly provided, it dynamically selects the
        best available device in the following order of priority:
        1. CUDA (if a GPU with CUDA support is available)
        2. MPS (Metal Performance Shaders, available on macOS with Apple Silicon)
        3. CPU (default fallback if no GPU is available)

        Args:
            device (str): The desired computation device.
            If not provided (None or an empty string), the method automatically
            determines the most suitable device.
        """
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

    def assign_chunking_strategy(self):
        """
        Assign a text chunking strategy based on the specified chunking strategy.

        This method initializes the appropriate text splitter based on the value
        of the `self.chunking_strategy` attribute.

        Currently, only supports "RecursiveCharacterTextSplitter" strategy.
        """
        if self.chunking_strategy == "RecursiveCharacterTextSplitter":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            raise ValueError(
                f"Unsupported chunking strategy: {self.chunking_strategy}. "
                "Currently, only 'RecursiveCharacterTextSplitter' is supported."
            )

    def initialize_model(self):
        """
        Initialize the embedding model and tokenizer.

        This method performs the following steps:
        - Sets the verbosity level of the Transformers library to suppress unnecessary warnings.
        - Determines model loading parameters based on the `mixed_precision` setting.
        - Loads the tokenizer associated with the embedding model.
        - Loads the embedding model with optional mixed precision settings.
        - Moves the model to the specified device (e.g., CPU, CUDA, MPS).
        - Sets the model to evaluation mode to disable training-specific layers like dropout.
        """
        tf_logging.set_verbosity_error()  # Suppress unnecessary warnings from the Transformers library.
        if self.mixed_precision:
            model_kwargs = {"torch_dtype": torch.float16}
        else:
            model_kwargs = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(
            self.embedding_model_name, **model_kwargs
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def create_identifier(self):
        """
        Generate a unique identifier for the current embedding configuration.

        This method creates a unique identifier based on the current timestamp,
        a randomly generated UUID, and key configuration parameters. The identifier
        can be used to uniquely tag output files or logs associated with the specific
        embedding run.

        The identifier is constructed in the following format:
        `<timestamp>_<uuid>_<model_name>_<chunking_strategy>_<chunk_size>_<chunk_overlap>_<max_length>`
        """

        unique_id = str(uuid.uuid4())
        time_stamp = datetime.datetime.now(pytz.timezone("US/Central")).strftime(
            "%m_%d_%y_%H_%M_%S"
        )
        model_name = os.path.basename(self.embedding_model_name)
        self.identifier = "{}_{}_{}_{}_{}_{}_{}".format(
            time_stamp,
            unique_id,
            model_name,
            self.chunking_strategy,
            self.chunk_size,
            self.chunk_overlap,
            self.max_length,
        )

    def create_output_dir(self, output_dir: str):
        """
        Create the specified output directory.
        If the directory already exists, it will be deleted and recreated.

        Args:
            output_dir (str): The relative path of the output directory to create.
        """
        self.output_dir = output_dir
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, self.output_dir)
        if os.path.exists(self.output_dir):
            print("Directory Path: {} exists, deleting it!".format(self.output_dir))
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        print("Created directory: {}!".format(self.output_dir))

    def setup_logging(self):
        """
        Set up logging configuration for the ingestion process.
        """
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

    def clear_gpu_memory(self):
        """
        Clear GPU memory cache.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

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

    def chunk_text(self, text: List[str]):
        """
        Splits a given text into smaller chunks using a splitter.

        Args:
            text (str): The input text to split.

        Returns:
            chunked_text (list[str]): A list of text chunks.
        """
        chunked_text = self.text_splitter.split_text(text)
        return chunked_text

    @autocast(device, enabled=(device != "cpu"))
    def generate_batch_embeddings(self, inputs: dict) -> torch.Tensor:
        """
        Generate embeddings for a single batch with mixed precision support

        Args:
            inputs (dict): A dictionary containing input tensors required by the model.
                - `input_ids` (torch.Tensor): Encoded input token IDs.
                - `attention_mask` (torch.Tensor): Binary mask indicating non-padding tokens.

        Returns:
            chunked_text (list[str]): A list of text chunks.
            embeddings (torch.Tensor): A tensor of L2-normalized sentence embeddings with shape (batch_size, embedding_dim).
        """
        outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = torch.sum(token_embeddings * input_mask, 1) / torch.clamp(
            input_mask.sum(1), min=1e-9
        )
        embeddings = normalize(embeddings, p=2, dim=1)
        return embeddings

    def generate_embeddings(self, chunked_text: List[str]) -> Iterator[np.ndarray]:
        """
        Generate embeddings for text chunks in batches with GPU memory management.

        Args:
            chunked_text (List[str]): List of text chunks

        Yields:
            np.ndarray: Embeddings for a single batch.
        """
        try:
            with torch.no_grad():
                for idx in range(0, len(chunked_text), self.batch_size):
                    batch_chunks = chunked_text[idx : idx + self.batch_size]
                    inputs = self.tokenizer(
                        batch_chunks,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    batch_embeddings = self.generate_batch_embeddings(inputs=inputs)
                    yield batch_embeddings.cpu().numpy()
                    if (idx + self.batch_size) % (self.batch_size * 10) == 0:
                        self.clear_gpu_memory()

        except RuntimeError as error:
            if "out of memory" in str(error):
                self.logger.error(
                    "GPU out of memory. Try reducing batch size or using CPU."
                )
                self.clear_gpu_memory()
            raise

    def create_embeddings(self, chunked_text: List[str]) -> np.ndarray:
        """
        Consolidate embeddings from batched into a single numpy array.

        Args:
            chunked_text (List[str]): List of text chunks

        Returns:
            np.ndarray: Consolidated embeddings array
        """
        embeddings = []
        for batch_embeddings in self.generate_embeddings(chunked_text=chunked_text):
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        return embeddings

    def ingest_document(self, file_path: str):
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
        help="File path of the document to be ingested.",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model supported by HuggingFace to create the embeddings.",
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
        help="The overlap window between two consecutive chunks.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=100,
        help="Batch size to generate the embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ingestion_output",
        help="Output directory to save the embeddings.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length to be used by the mdoel",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether to use Mixed Precision Inference ('yes' or 'no')",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Whether to use 'cuda', 'mps' or 'cpu' for acceleration ('cuda', 'mps' or 'cpu')",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(
        "\n-------------------------------------\nIngesting the document!\n-------------------------------------\n"
    )
    args = get_args()
    print("Arguments: {}".format(vars(args)))

    if args.accelerator:
        device = args.accelerator

    # context = flywheel.GearContext()
    # config = context.config

    embedder = IngestDocument(
        embedding_model_name=args.embedding_model_name,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        output_dir=args.output_dir,
        device=device,
        max_length=args.max_length,
        mixed_precision=bool(args.mixed_precision),
    )

    embedder.logger.info("-------------------------------------")
    embedder.logger.info("Ingesting the document!")
    embedder.logger.info("-------------------------------------")
    embedder.logger.info("Arguments: {}".format(vars(args)))
    embedder.logger.info(
        "Initialized model: {} on device: {}!".format(
            embedder.embedding_model_name, embedder.device
        )
    )

    # embedder = IngestDocument(
    #     embedding_model_name=config["embedding_model_name"],
    #     chunking_strategy=config["chunking_strategy"],
    #     chunk_size=config["chunk_size"],
    #     chunk_overlap=config["chunk_overlap"],
    #     embedding_batch_size=config["embedding_batch_size"],
    #     output_dir=config["output_dir"],
    #     device=device,
    #     max_length=config["max_length"],
    #     mixed_precision=bool(config["mixed_precision"])
    # )

    embedder.ingest_document(file_path=args.file_path)
    # embedder.ingest_document(file_path=context.get_input_path("document_file"))

    embedder.logger.info("-------------------------------------")
    embedder.logger.info("Document has been ingested successfully!")
    embedder.logger.info("-------------------------------------")

    print(
        "\n-------------------------------------\nDocument has been ingested successfully!\n-------------------------------------\n"
    )
