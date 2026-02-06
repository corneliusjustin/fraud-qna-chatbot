import json
import logging
from models.enums import QueryType
from models.schemas import ClassificationResult
from services.together_ai import chat_completion_routing

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a query classifier for a fraud analysis system. Classify the user's question into one of three categories:

1. "sql" - Questions about statistical data, counts, rates, trends, aggregations, or specific transaction data from the fraud_transactions database table. Examples: fraud rates over time, merchant categories with most fraud, average amounts, geographic patterns.

2. "rag" - Questions about concepts, methods, definitions, explanations, or qualitative information from a document about credit card fraud. Examples: what are the methods of fraud, what are the components of a detection system, how does fraud work.

3. "hybrid" - Questions that need BOTH statistical data AND document knowledge. Examples: comparing dataset statistics with document claims, questions about specific report statistics (like EEA, H1 2023, cross-border).

Respond with ONLY a JSON object (no markdown, no code blocks):
{
    "query_type": "sql" or "rag" or "hybrid",
    "reasoning": "brief explanation",
    "sql_query_hint": "what to query if sql is needed, or null",
    "rag_search_hint": "what to search if rag is needed, or null"
}
"""


def classify_query(question: str, history: list[dict] | None = None) -> ClassificationResult:
    messages = [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
    ]
    # Include recent conversation history for context on follow-up questions
    if history:
        for msg in history[-6:]:  # last 3 exchanges
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        raw = chat_completion_routing(messages)
        
        # Clean up response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        data = json.loads(raw)
        
        query_type = data.get("query_type", "unknown")
        if query_type not in ["sql", "rag", "hybrid"]:
            query_type = "unknown"

        return ClassificationResult(
            query_type=QueryType(query_type),
            reasoning=data.get("reasoning", ""),
            sql_query_hint=data.get("sql_query_hint"),
            rag_search_hint=data.get("rag_search_hint"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classification response: {e}. Raw: {raw}")
        return _fallback_classification(question)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return _fallback_classification(question)


def _fallback_classification(question: str) -> ClassificationResult:
    q = question.lower()
    
    sql_keywords = [
        "how many", "count", "rate", "trend", "average", "total",
        "highest", "lowest", "most", "least", "percentage", "monthly",
        "daily", "yearly", "over time", "fluctuate", "merchant", "category",
        "transaction", "amount", "which", "top", "statistics",
    ]
    rag_keywords = [
        "what are", "explain", "describe", "methods", "components",
        "according to", "authors", "definition", "how does", "why",
        "primary methods", "core components", "detection system",
        "techniques", "strategies",
    ]
    hybrid_keywords = [
        "eea", "cross-border", "h1 2023", "report", "compared to",
        "outside the", "share of total",
    ]

    sql_score = sum(1 for kw in sql_keywords if kw in q)
    rag_score = sum(1 for kw in rag_keywords if kw in q)
    hybrid_score = sum(1 for kw in hybrid_keywords if kw in q)

    if hybrid_score > 0:
        qt = QueryType.HYBRID
    elif sql_score > rag_score:
        qt = QueryType.SQL
    elif rag_score > 0:
        qt = QueryType.RAG
    else:
        qt = QueryType.HYBRID

    return ClassificationResult(
        query_type=qt,
        reasoning="Fallback keyword-based classification",
        sql_query_hint=question if qt in [QueryType.SQL, QueryType.HYBRID] else None,
        rag_search_hint=question if qt in [QueryType.RAG, QueryType.HYBRID] else None,
    )
