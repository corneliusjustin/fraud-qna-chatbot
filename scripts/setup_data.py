import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_all():
    logger.info("=" * 60)
    logger.info("Starting data setup...")
    logger.info("=" * 60)

    # Step 1: Setup SQLite database
    logger.info("\n[1/2] Setting up SQLite database...")
    from services.database import setup_database, validate_database

    db_stats = setup_database()
    logger.info(f"Database stats: {db_stats}")

    validation = validate_database()
    logger.info(f"Database validation: {validation}")

    # Step 2: Setup ChromaDB vector store
    logger.info("\n[2/2] Setting up ChromaDB vector store...")
    from tools.document_processor import process_pdf
    from services.vector_store import add_documents, validate_vector_store

    chunks = process_pdf()
    logger.info(f"Processed {len(chunks)} chunks from PDF")

    doc_count = add_documents(chunks)
    logger.info(f"Documents in vector store: {doc_count}")

    vs_validation = validate_vector_store()
    logger.info(f"Vector store validation: {vs_validation}")

    logger.info("\n" + "=" * 60)
    logger.info("Data setup complete!")
    logger.info(f"  SQLite: {validation['total_rows']} rows ({validation['fraud_count']} fraud)")
    logger.info(f"  ChromaDB: {vs_validation['total_chunks']} chunks")
    logger.info("=" * 60)

    return {
        "database": validation,
        "vector_store": vs_validation,
    }


if __name__ == "__main__":
    setup_all()
