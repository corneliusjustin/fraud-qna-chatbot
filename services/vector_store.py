import logging
from pathlib import Path
import chromadb
from services.together_ai import get_embeddings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

_client = None
_collection = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="fraud_reports",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(chunks: list[dict]) -> int:
    collection = get_collection()

    if collection.count() > 0:
        logger.info(f"Collection already has {collection.count()} documents, skipping ingestion")
        return collection.count()

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Pre-compute embeddings with Together AI
    logger.info(f"Computing embeddings for {len(documents)} chunks...")
    embeddings = get_embeddings(documents)
    logger.info(f"Computed {len(embeddings)} embeddings")

    # Add in batches with pre-computed embeddings
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )
        logger.info(f"Added batch {i // batch_size + 1}: chunks {i} to {end - 1}")

    logger.info(f"Total documents in collection: {collection.count()}")
    return collection.count()


def search_documents(query: str, n_results: int = 7) -> dict:
    collection = get_collection()
    if collection.count() == 0:
        return {"documents": [], "metadatas": [], "distances": []}

    # Pre-compute query embedding
    query_embedding = get_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )

    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def is_vector_store_ready() -> bool:
    try:
        collection = get_collection()
        return collection.count() > 0
    except Exception:
        return False


def validate_vector_store() -> dict:
    collection = get_collection()
    count = collection.count()

    test_results = None
    if count > 0:
        test_results = search_documents("fraud detection methods", n_results=3)

    return {
        "total_chunks": count,
        "test_query_results": len(test_results["documents"]) if test_results else 0,
    }
