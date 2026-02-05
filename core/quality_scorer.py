import json
import logging
from models.schemas import QualityScore, SQLResult, RAGResult
from services.together_ai import chat_completion_routing

logger = logging.getLogger(__name__)

SCORING_PROMPT = """You are a quality assurance evaluator for an AI fraud analysis system. Evaluate the quality of the given answer against the source context.

SCORING RUBRIC (1-5):
5 = Fully accurate, cites specific data/pages, answers all parts of the question
4 = Accurate with minor omissions, good citations
3 = Mostly accurate, answers the core question but may lack detail or citations
2 = Partially correct, missing key information or contains unsupported claims
1 = Incorrect, hallucinated, or fails to address the question

EVALUATION CRITERIA:
1. Faithfulness: Does the answer only contain information from the provided context?
2. Completeness: Does it answer all parts of the question?
3. Citations: Does it reference specific data points, pages, or sources?
4. Accuracy: Are all numbers and claims verifiable from the context?

CONTEXT:
{context}

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

Respond with ONLY a JSON object (no markdown, no code blocks):
{{
    "score": <1-5>,
    "reasoning": "explanation of the score",
    "has_hallucination": true/false,
    "missing_information": ["list of missing items"] or []
}}
"""


def score_response(
    question: str,
    answer: str,
    sql_result: SQLResult | None = None,
    rag_result: RAGResult | None = None,
) -> QualityScore:
    # Build context summary for scoring
    context_parts = []

    if sql_result and not sql_result.error and sql_result.rows:
        rows_preview = sql_result.rows[:10]
        context_parts.append(
            f"SQL Query: {sql_result.query}\n"
            f"Columns: {sql_result.columns}\n"
            f"Sample rows: {rows_preview}\n"
            f"Total rows: {sql_result.row_count}"
        )

    if rag_result and not rag_result.error and rag_result.chunks:
        for chunk, meta in zip(rag_result.chunks[:5], rag_result.metadatas[:5]):
            page = meta.get("page_number", "?")
            context_parts.append(f"[Page {page}]: {chunk[:500]}")

    if not context_parts:
        return QualityScore(
            score=2,
            reasoning="No source context available for verification",
            has_hallucination=False,
            missing_information=["No context data to verify against"],
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": SCORING_PROMPT.format(
                context=context,
                question=question,
                answer=answer,
            ),
        },
        {"role": "user", "content": "Evaluate the answer quality."},
    ]

    try:
        raw = chat_completion_routing(messages)
        
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        data = json.loads(raw)

        score = max(1, min(5, int(data.get("score", 3))))
        return QualityScore(
            score=score,
            reasoning=data.get("reasoning", ""),
            has_hallucination=bool(data.get("has_hallucination", False)),
            missing_information=data.get("missing_information", []),
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse quality score: {e}")
        return QualityScore(
            score=3,
            reasoning="Could not parse quality evaluation; defaulting to moderate score",
            has_hallucination=False,
            missing_information=[],
        )
    except Exception as e:
        logger.error(f"Quality scoring error: {e}")
        return QualityScore(
            score=3,
            reasoning=f"Scoring error: {str(e)}",
            has_hallucination=False,
            missing_information=[],
        )
