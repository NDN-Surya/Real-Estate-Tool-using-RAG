# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

# Load environment variables (if any)
load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

# Global variables for LLM and Vector Store
llm = None
vector_store = None


def initialize_components():
    """
    Initializes the language model and vector store.
    Configures Chroma to use PostgreSQL as the backend for persistence.
    """
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        # Configure Chroma to use PostgreSQL
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            client_settings=Settings(
                chroma_api_impl="postgresql",
                chroma_db_impl="postgresql",
                postgres_user="your_user",  # Replace with your PostgreSQL username
                postgres_password="your_password",  # Replace with your PostgreSQL password
                postgres_host="localhost",  # Replace with your PostgreSQL host
                postgres_port="5432",  # Replace with your PostgreSQL port
                postgres_database="your_database",  # Replace with your PostgreSQL database name
                anonymized_telemetry=False
            )
        )


def process_urls(urls):
    """
    Processes the provided URLs, scrapes data, splits it into chunks,
    and stores the chunks in the vector database.

    :param urls: List of URLs to process
    :yield: Status messages during processing
    """
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"


def generate_answer(query):
    """
    Generates an answer to the given query using the vector store.

    :param query: User's question
    :return: Tuple containing the answer and sources
    :raises RuntimeError: If vector store is not initialized
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources


if __name__ == "__main__":
    # Example URLs for testing
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    # Process the URLs and print the status
    for status in process_urls(urls):
        print(status)

    # Example query and output
    try:
        answer, sources = generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date?")
        print(f"Answer: {answer}")
        print(f"Sources: {sources}")
    except RuntimeError as e:
        print(f"Error: {e}")

