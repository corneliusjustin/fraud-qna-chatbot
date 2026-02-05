import logging
from collections.abc import Generator
from dataclasses import dataclass
from models.schemas import AgentResponse, SQLResult, RAGResult, QualityScore
from models.enums import QueryType
from core.query_classifier import classify_query
from core.response_synthesizer import synthesize_response, synthesize_response_stream
from core.quality_scorer import score_response
from tools.sql_tool import run_sql_query
from tools.rag_tool import search_docs
from utils.helpers import sanitize_input
from utils.error_handler import handle_llm_error, handle_sql_error, handle_rag_error

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 3
MAX_RETRIES = 2


@dataclass
class AgentStep:
    step: str       # e.g. "classify", "sql", "rag", "synthesize", "score", "retry", "done", "error"
    label: str      # human-readable status text
    detail: str = ""


def process_query(question: str) -> AgentResponse:
    # Non-streaming version (used by tests)
    result = None
    for event in process_query_stream(question):
        if isinstance(event, AgentResponse):
            result = event
    return result


def process_query_stream(question: str) -> Generator[AgentStep | AgentResponse | str, None, None]:
    question = sanitize_input(question)
    if not question:
        yield AgentResponse(
            answer="Please enter a question to get started.",
            query_type=QueryType.UNKNOWN,
            error="Empty query",
        )
        return

    try:
        # Step 1: Classify
        yield AgentStep("classify", "🔍 Classifying your question...")
        classification = classify_query(question)
        logger.info(f"Classification: {classification.query_type} - {classification.reasoning}")

        type_labels = {
            QueryType.SQL: "📊 Statistical query — will query the database",
            QueryType.RAG: "📄 Document query — will search the PDF",
            QueryType.HYBRID: "📊📄 Hybrid query — will use both database and PDF",
        }
        yield AgentStep(
            "classify_done",
            type_labels.get(classification.query_type, "❓ Unknown query type"),
            classification.reasoning,
        )

        best_response = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                yield AgentStep("retry", f"🔄 Retrying (attempt {attempt + 1}/{MAX_RETRIES + 1})...")

            sql_result = None
            rag_result = None

            # Step 2a: SQL tool
            if classification.query_type in [QueryType.SQL, QueryType.HYBRID]:
                yield AgentStep("sql", "📊 Generating and executing SQL query...")
                try:
                    sql_result = run_sql_query(classification.sql_query_hint or question)
                    if sql_result.error:
                        logger.warning(f"SQL tool error: {sql_result.error}")
                        yield AgentStep("sql_done", f"⚠️ SQL query issue: {sql_result.error[:80]}")
                    else:
                        yield AgentStep("sql_done", f"✅ SQL returned {sql_result.row_count} rows", sql_result.query)
                except Exception as e:
                    logger.error(f"SQL tool exception: {e}")
                    sql_result = SQLResult(query="", error=handle_sql_error(e))
                    yield AgentStep("sql_done", f"❌ SQL error: {str(e)[:80]}")

            # Step 2b: RAG tool
            if classification.query_type in [QueryType.RAG, QueryType.HYBRID]:
                yield AgentStep("rag", "📄 Searching document for relevant information...")
                try:
                    rag_result = search_docs(classification.rag_search_hint or question)
                    if rag_result.error:
                        logger.warning(f"RAG tool error: {rag_result.error}")
                        yield AgentStep("rag_done", f"⚠️ Document search issue: {rag_result.error[:80]}")
                    else:
                        pages = sorted({m.get("page_number", "?") for m in rag_result.metadatas})
                        yield AgentStep("rag_done", f"✅ Found {len(rag_result.chunks)} relevant chunks (pages {', '.join(str(p) for p in pages)})")
                except Exception as e:
                    logger.error(f"RAG tool exception: {e}")
                    rag_result = RAGResult(error=handle_rag_error(e))
                    yield AgentStep("rag_done", f"❌ RAG error: {str(e)[:80]}")

            # Graceful degradation
            if classification.query_type == QueryType.HYBRID:
                if sql_result and sql_result.error and rag_result and not rag_result.error:
                    logger.info("Hybrid degraded to RAG-only due to SQL failure")
                elif rag_result and rag_result.error and sql_result and not sql_result.error:
                    logger.info("Hybrid degraded to SQL-only due to RAG failure")

            # Step 3: Synthesize (streaming)
            yield AgentStep("synthesize", "✨ Generating response...")

            try:
                answer_chunks = []
                for token in synthesize_response_stream(
                    question=question,
                    query_type=classification.query_type,
                    sql_result=sql_result,
                    rag_result=rag_result,
                ):
                    answer_chunks.append(token)
                    yield token  # stream token to UI
                answer = "".join(answer_chunks)
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                answer = handle_llm_error(e)
                yield answer  # yield the error as the full text

            # Step 4: Quality scoring
            yield AgentStep("score", "🔍 Evaluating response quality...")
            try:
                quality = score_response(
                    question=question,
                    answer=answer,
                    sql_result=sql_result,
                    rag_result=rag_result,
                )
            except Exception as e:
                logger.error(f"Quality scoring error: {e}")
                quality = QualityScore(
                    score=3,
                    reasoning=f"Scoring failed: {str(e)}",
                    has_hallucination=False,
                    missing_information=[],
                )

            score_emoji = "🟢" if quality.score >= 4 else "🟡" if quality.score >= 3 else "🔴"
            yield AgentStep("score_done", f"{score_emoji} Quality score: {quality.score}/5", quality.reasoning)

            sources = _build_sources(sql_result, rag_result)

            current_response = AgentResponse(
                answer=answer,
                query_type=classification.query_type,
                sql_result=sql_result,
                rag_result=rag_result,
                quality_score=quality,
                sources=sources,
                retry_count=attempt,
            )

            if best_response is None or quality.score > (best_response.quality_score.score if best_response.quality_score else 0):
                best_response = current_response

            if quality.score >= QUALITY_THRESHOLD:
                logger.info(f"Quality score {quality.score} meets threshold on attempt {attempt + 1}")
                yield current_response
                return

            logger.warning(f"Quality score {quality.score} below threshold on attempt {attempt + 1}")

        # Return best after retries exhausted
        if best_response and best_response.quality_score and best_response.quality_score.score < QUALITY_THRESHOLD:
            best_response.answer += (
                "\n\n---\n*Note: This response may have limited accuracy. "
                "The quality score did not meet the confidence threshold after multiple attempts. "
                "Please verify the information against the source data.*"
            )

        yield best_response or current_response

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        yield AgentStep("error", f"❌ Error: {str(e)}")
        yield AgentResponse(
            answer=f"An unexpected error occurred: {str(e)}. Please try again.",
            query_type=QueryType.UNKNOWN,
            error=str(e),
        )


def _build_sources(sql_result: SQLResult | None, rag_result: RAGResult | None) -> list[str]:
    sources = []
    if sql_result and not sql_result.error and sql_result.rows:
        sources.append(f"SQL: {sql_result.query}")
    if rag_result and not rag_result.error and rag_result.chunks:
        pages = set()
        for meta in rag_result.metadatas:
            page = meta.get("page_number")
            if page:
                pages.add(page)
        if pages:
            page_list = sorted(pages)
            sources.append(f"Document: Understanding Credit Card Frauds (Pages {', '.join(str(p) for p in page_list)})")
    return sources
