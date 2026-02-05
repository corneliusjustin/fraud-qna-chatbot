import logging
from services.vector_store import search_documents, is_vector_store_ready
from models.schemas import RAGResult

logger = logging.getLogger(__name__)


def search_docs(query: str, n_results: int = 7) -> RAGResult:
    if not is_vector_store_ready():
        return RAGResult(error="Vector store is not initialized. Please run data setup first.")

    try:
        results = search_documents(query, n_results=n_results)
        
        chunks = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        if not chunks:
            return RAGResult(
                chunks=[],
                metadatas=[],
                distances=[],
                error="No relevant documents found for this query.",
            )
        
        return RAGResult(
            chunks=chunks,
            metadatas=metadatas,
            distances=distances,
        )

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return RAGResult(error=f"Document retrieval failed: {str(e)}")


def format_rag_context(rag_result: RAGResult) -> str:
    if rag_result.error or not rag_result.chunks:
        return ""

    context_parts = []
    for i, (chunk, meta) in enumerate(zip(rag_result.chunks, rag_result.metadatas)):
        page = meta.get("page_number", "?")
        source = meta.get("source", "unknown")
        context_parts.append(
            f"[Source: {source}, Page {page}]\n{chunk}"
        )

    return "\n\n---\n\n".join(context_parts)
