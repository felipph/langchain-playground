from dotenv import load_dotenv
import os

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
load_dotenv()


URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
        port=os.environ.get("PGVECTORS_PORT"),
        host=os.environ.get("PGVECTORS_HOST"),
        username=os.environ.get("PGVECTORS_USER"),
        password=os.environ.get("PGVECTORS_PASS"),
        db_name=os.environ.get("PGVECTORS_DB_NAME"),
    )
embeddings = OllamaEmbeddings(model="nomic-embed-text")
collection_name = "medium_blogs_embbedings"
vectorstore = PGVecto_rs.from_collection_name(
    embedding=embeddings,
    db_url=URL,
    # The table name is f"collection_{collection_name}", so that it should be unique.
    collection_name=collection_name,
)