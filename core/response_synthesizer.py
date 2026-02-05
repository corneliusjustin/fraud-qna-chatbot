import logging
from collections.abc import Generator
from models.schemas import SQLResult, RAGResult
from models.enums import QueryType
from services.together_ai import chat_completion, chat_completion_stream
from utils.helpers import format_sql_result_as_text

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are an expert fraud analyst. Synthesize a comprehensive, accurate answer based ONLY on the provided context data.

RULES:
1. Base your answer STRICTLY on the provided context. Do NOT add information not present in the context.
2. If SQL data is provided, reference specific numbers, percentages, and trends.
3. If document text is provided, cite the page number in brackets like [Page X].
4. If both are provided, integrate them coherently.
5. Structure your response clearly with headings or bullet points when appropriate.
6. If the context is insufficient to fully answer the question, explicitly state what information is missing.
7. For time-series data, describe the trend (increasing, decreasing, seasonal patterns).
8. Round percentages to 2 decimal places.

{context_section}

Answer the following question thoroughly and accurately:
"""


def synthesize_response(
    question: str,
    query_type: QueryType,
    sql_result: SQLResult | None = None,
    rag_result: RAGResult | None = None,
) -> str:
    context_parts = []

    if sql_result and not sql_result.error and sql_result.rows:
        table_text = format_sql_result_as_text(sql_result.columns, sql_result.rows)
        context_parts.append(
            f"## SQL Query Results\n"
            f"Query: {sql_result.query}\n"
            f"Rows returned: {sql_result.row_count}\n\n"
            f"{table_text}"
        )

    if rag_result and not rag_result.error and rag_result.chunks:
        doc_parts = []
        for i, (chunk, meta) in enumerate(zip(rag_result.chunks, rag_result.metadatas)):
            page = meta.get("page_number", "?")
            source = meta.get("source", "unknown")
            doc_parts.append(f"[Source: {source}, Page {page}]\n{chunk}")
        context_parts.append(
            f"## Document Context\n" + "\n\n---\n\n".join(doc_parts)
        )

    if not context_parts:
        return _handle_no_context(question, query_type, sql_result, rag_result)

    context_section = "\n\n".join(context_parts)
    messages = [
        {
            "role": "system",
            "content": SYNTHESIS_PROMPT.format(context_section=f"CONTEXT:\n{context_section}"),
        },
        {"role": "user", "content": question},
    ]

    return chat_completion(messages, max_tokens=3000)


def build_synthesis_messages(
    question: str,
    query_type: QueryType,
    sql_result: SQLResult | None = None,
    rag_result: RAGResult | None = None,
) -> list[dict] | None:
    context_parts = []

    if sql_result and not sql_result.error and sql_result.rows:
        table_text = format_sql_result_as_text(sql_result.columns, sql_result.rows)
        context_parts.append(
            f"## SQL Query Results\n"
            f"Query: {sql_result.query}\n"
            f"Rows returned: {sql_result.row_count}\n\n"
            f"{table_text}"
        )

    if rag_result and not rag_result.error and rag_result.chunks:
        doc_parts = []
        for i, (chunk, meta) in enumerate(zip(rag_result.chunks, rag_result.metadatas)):
            page = meta.get("page_number", "?")
            source = meta.get("source", "unknown")
            doc_parts.append(f"[Source: {source}, Page {page}]\n{chunk}")
        context_parts.append(
            f"## Document Context\n" + "\n\n---\n\n".join(doc_parts)
        )

    if not context_parts:
        return None

    context_section = "\n\n".join(context_parts)
    return [
        {
            "role": "system",
            "content": SYNTHESIS_PROMPT.format(context_section=f"CONTEXT:\n{context_section}"),
        },
        {"role": "user", "content": question},
    ]


def synthesize_response_stream(
    question: str,
    query_type: QueryType,
    sql_result: SQLResult | None = None,
    rag_result: RAGResult | None = None,
) -> Generator[str, None, None]:
    messages = build_synthesis_messages(question, query_type, sql_result, rag_result)
    if messages is None:
        yield _handle_no_context(question, query_type, sql_result, rag_result)
        return
    yield from chat_completion_stream(messages, max_tokens=3000)


def _handle_no_context(
    question: str,
    query_type: QueryType,
    sql_result: SQLResult | None,
    rag_result: RAGResult | None,
) -> str:
    errors = []
    if sql_result and sql_result.error:
        errors.append(f"SQL Tool: {sql_result.error}")
    if rag_result and rag_result.error:
        errors.append(f"RAG Tool: {rag_result.error}")

    if errors:
        return (
            "I was unable to retrieve the necessary data to answer your question.\n\n"
            "**Issues encountered:**\n" +
            "\n".join(f"- {e}" for e in errors) +
            "\n\nPlease try rephrasing your question or check that the data sources are properly set up."
        )

    if sql_result and sql_result.row_count == 0:
        return (
            "The database query returned no results for your question. "
            "This might mean the data doesn't contain matching records. "
            "Try broadening your query or asking about different criteria."
        )

    return "I couldn't find relevant information to answer your question. Please try rephrasing it."
