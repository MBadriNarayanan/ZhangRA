import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_data(file_type, file_path):
    data = None
    if file_type == "csv":
        loader = CSVLoader(file_path=file_path)
        data = loader.load()
    return data


def split_data(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    data_split = text_splitter.split_documents(data)
    return data_split


def create_vector_store(embedding_model_name, data_split):
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    vector_store = Chroma.from_documents(documents=data_split, embedding=embeddings)

    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_llm(model_name, model_temperature, max_tokens):
    return ChatOllama(
        model=model_name, temperature=model_temperature, max_tokens=max_tokens
    )


def prepare_data(args, logger):
    try:
        data = load_data(file_type=args.file_type, file_path=args.file_path)
        data_split = split_data(
            data, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        vector_store = create_vector_store(
            embedding_model_name=args.embedding_model_name, data_split=data_split
        )
        retriever = vector_store.as_retriever(
            search_type=args.search_type, search_kwargs={"k": args.top_n}
        )
        logger.info(
            "Data has been stored in a vector database and retriever has been created!"
        )
        logger.info("-------------------------------------")
        print(
            "Data has been stored in a vector database and retriever has been created!"
        )
        print("-------------------------------------")
        return retriever
    except Exception as error:
        logger.error("Error while creating the vector database: {}".format(error))
        logger.error("-------------------------------------")
        print("Error while creating the vector database: {}".format(error))
        print("-------------------------------------")
        sys.exit()


def create_chatbot(args, logger):
    try:
        llm = load_llm(
            model_name=args.model_name,
            model_temperature=args.model_temperature,
            max_tokens=args.max_tokens,
        )
        retriever = prepare_data(args, logger)

        prompt = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(prompt)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG Chain has been created!")
        logger.info("-------------------------------------")
        print("RAG Chain has been created!")
        print("-------------------------------------")
        return rag_chain
    except Exception as error:
        logger.error("Error while creating the RAG Chain: {}".format(error))
        logger.error("-------------------------------------")
        print("Error while creating the RAG Chain: {}".format(error))
        print("-------------------------------------")
        sys.exit()


def run_chatbot(args, logger):
    try:
        rag_chain = create_chatbot(args=args, logger=logger)
        logger.info("Welcome user, this is an LLM powered chatbot!")
        print("Welcome user, this is an LLM powered chatbot!")
        choice = True
        while choice:
            question = input("Enter your question: ")
            logger.info("Question: {}".format(question))
            print("Question: {}".format(question))

            answer = rag_chain.invoke(question)
            answer = answer.replace("Answer: ", "")
            print("Answer: {}".format(answer))
            logger.info("Answer: {}".format(answer))
            choice = input("Do you have more questions (Y / Yes / N / No): ")
            choice = choice.lower()
            if choice in ["n", "no"]:
                logger.info("Choice: No")
                logger.info("Exiting the chatbot!")
                logger.info("-------------------------------------")
                print("Choice: No")
                print("Exiting the chatbot!")
                print("-------------------------------------")
                choice = False
            else:
                logger.info("Choice: Yes")
                logger.info("Continuing to the next question!")
                logger.info("-------------------------------------")
                print("Choice: Yes")
                print("Continuing to the next question!")
                print("-------------------------------------")
                choice = True

    except Exception as error:
        logger.error("Error while running the chatbot: {}".format(error))
        logger.error("-------------------------------------")
        print("Error while running the chatbot: {}".format(error))
        print("-------------------------------------")
        sys.exit()
