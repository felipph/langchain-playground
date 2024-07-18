import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
load_dotenv()

if __name__ == '__main__':
    print("Iniciando a ingestão")
    loader = TextLoader("mediumblog1.txt", encoding="utf8")

    documents = loader.load()

    print("Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} parts")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")


# See docker command above to launch a postgres instance with pgvector enabled.
    collection_name = "medium_blogs_embbedings"

    URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
        port=os.environ.get("PGVECTORS_PORT"),
        host=os.environ.get("PGVECTORS_HOST"),
        username=os.environ.get("PGVECTORS_USER"),
        password=os.environ.get("PGVECTORS_PASS"),
        db_name=os.environ.get("PGVECTORS_DB_NAME"),
    )

    vectorstore = PGVecto_rs.from_documents(
        documents=texts,
        embedding=embeddings,
        db_url=URL,
        # The table name is f"collection_{collection_name}", so that it should be unique.
        collection_name=collection_name,
    )
