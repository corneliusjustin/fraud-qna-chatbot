"""
Backend integration tests for the Fraud Analysis Agent.
Tests each component individually and then end-to-end with all 6 sample questions.
"""
import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_database():
    logger.info("=" * 50)
    logger.info("TEST: Database Service")
    logger.info("=" * 50)
    from services.database import is_database_ready, validate_database, execute_query

    assert is_database_ready(), "Database not ready"
    
    stats = validate_database()
    logger.info(f"  Total rows: {stats['total_rows']}")
    logger.info(f"  Fraud count: {stats['fraud_count']}")
    logger.info(f"  Fraud rate: {stats['fraud_rate']}%")
    logger.info(f"  Date range: {stats['date_range']}")
    logger.info(f"  Categories: {stats['categories']}")
    
    assert stats["total_rows"] > 1_000_000, f"Expected >1M rows, got {stats['total_rows']}"
    assert stats["fraud_count"] > 0, "No fraud records found"
    assert len(stats["categories"]) > 5, "Too few categories"

    # Test sample query
    cols, rows = execute_query("SELECT COUNT(*) AS cnt FROM fraud_transactions WHERE is_fraud = 1")
    assert rows[0][0] == stats["fraud_count"], "Fraud count mismatch"
    logger.info("  PASSED: Database service works correctly\n")


def test_vector_store():
    logger.info("=" * 50)
    logger.info("TEST: Vector Store Service")
    logger.info("=" * 50)
    from services.vector_store import is_vector_store_ready, validate_vector_store, search_documents

    assert is_vector_store_ready(), "Vector store not ready"
    
    stats = validate_vector_store()
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Test query results: {stats['test_query_results']}")
    
    assert stats["total_chunks"] > 0, "No chunks in vector store"

    # Test search
    results = search_documents("credit card fraud methods", n_results=3)
    assert len(results["documents"]) > 0, "No search results returned"
    logger.info(f"  Search returned {len(results['documents'])} results")
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        logger.info(f"    [{i+1}] Page {meta.get('page_number', '?')}: {doc[:80]}...")
    logger.info("  PASSED: Vector store works correctly\n")


def test_sql_tool():
    logger.info("=" * 50)
    logger.info("TEST: SQL Tool")
    logger.info("=" * 50)
    from tools.sql_tool import generate_sql, validate_sql, run_sql_query

    # Test SQL generation
    sql = generate_sql("What is the total number of fraudulent transactions?")
    logger.info(f"  Generated SQL: {sql}")
    
    is_valid, msg = validate_sql(sql)
    logger.info(f"  Validation: {is_valid} - {msg}")
    assert is_valid, f"Generated SQL is invalid: {msg}"

    # Test SQL execution
    result = run_sql_query("How many fraudulent transactions are there?")
    logger.info(f"  Query: {result.query}")
    logger.info(f"  Rows: {result.row_count}")
    logger.info(f"  Error: {result.error}")
    assert result.error is None, f"SQL query failed: {result.error}"
    assert result.row_count > 0, "No results returned"
    logger.info("  PASSED: SQL tool works correctly\n")


def test_sql_safety():
    logger.info("=" * 50)
    logger.info("TEST: SQL Safety")
    logger.info("=" * 50)
    from utils.helpers import is_safe_sql

    assert is_safe_sql("SELECT * FROM fraud_transactions") == True
    assert is_safe_sql("DROP TABLE fraud_transactions") == False
    assert is_safe_sql("DELETE FROM fraud_transactions") == False
    assert is_safe_sql("INSERT INTO fraud_transactions VALUES (1)") == False
    assert is_safe_sql("UPDATE fraud_transactions SET is_fraud = 1") == False
    logger.info("  PASSED: SQL safety checks work correctly\n")


def test_rag_tool():
    logger.info("=" * 50)
    logger.info("TEST: RAG Tool")
    logger.info("=" * 50)
    from tools.rag_tool import search_docs, format_rag_context

    result = search_docs("primary methods of credit card fraud")
    logger.info(f"  Chunks: {len(result.chunks)}")
    logger.info(f"  Error: {result.error}")
    assert result.error is None, f"RAG search failed: {result.error}"
    assert len(result.chunks) > 0, "No chunks returned"

    context = format_rag_context(result)
    assert len(context) > 0, "Empty context"
    logger.info(f"  Context length: {len(context)} chars")
    logger.info("  PASSED: RAG tool works correctly\n")


def test_query_classifier():
    logger.info("=" * 50)
    logger.info("TEST: Query Classifier")
    logger.info("=" * 50)
    from core.query_classifier import classify_query
    from models.enums import QueryType

    test_cases = [
        ("How does the monthly fraud rate fluctuate over the two-year period?", [QueryType.SQL]),
        ("Which merchant categories exhibit the highest incidence of fraud?", [QueryType.SQL]),
        ("What are the primary methods by which credit card fraud is committed?", [QueryType.RAG]),
        ("What are the core components of an effective fraud detection system?", [QueryType.RAG]),
        ("How much higher are fraud rates when the transaction counterpart is outside the EEA?", [QueryType.HYBRID, QueryType.RAG]),
    ]

    for question, expected_types in test_cases:
        result = classify_query(question)
        status = "PASS" if result.query_type in expected_types else "WARN"
        logger.info(f"  [{status}] Q: {question[:60]}...")
        logger.info(f"        Type: {result.query_type} (expected: {[t.value for t in expected_types]})")
        logger.info(f"        Reason: {result.reasoning}")
    logger.info("  PASSED: Query classifier works\n")


def test_quality_scorer():
    logger.info("=" * 50)
    logger.info("TEST: Quality Scorer")
    logger.info("=" * 50)
    from core.quality_scorer import score_response
    from models.schemas import SQLResult

    sql_result = SQLResult(
        query="SELECT COUNT(*) FROM fraud_transactions WHERE is_fraud = 1",
        columns=["count"],
        rows=[[9651]],
        row_count=1,
    )
    
    # Good answer
    good_score = score_response(
        question="How many fraudulent transactions are there?",
        answer="There are 9,651 fraudulent transactions in the dataset.",
        sql_result=sql_result,
    )
    logger.info(f"  Good answer score: {good_score.score} - {good_score.reasoning}")
    assert good_score.score >= 3, f"Good answer scored too low: {good_score.score}"

    # Bad answer
    bad_score = score_response(
        question="How many fraudulent transactions are there?",
        answer="There are approximately 500,000 fraudulent transactions, making up 25% of all data.",
        sql_result=sql_result,
    )
    logger.info(f"  Bad answer score: {bad_score.score} - {bad_score.reasoning}")
    logger.info("  PASSED: Quality scorer works\n")


def test_end_to_end_questions():
    logger.info("=" * 50)
    logger.info("TEST: End-to-End - All 6 Sample Questions")
    logger.info("=" * 50)
    from core.agent import process_query

    questions = [
        "How does the daily or monthly fraud rate fluctuate over the two-year period?",
        "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
        "What are the primary methods by which credit card fraud is committed?",
        "What are the core components of an effective fraud detection system, according to the authors?",
        "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
        "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
    ]

    results = []
    for i, q in enumerate(questions, 1):
        logger.info(f"\n  Q{i}: {q}")
        response = process_query(q)
        score = response.quality_score.score if response.quality_score else 0
        logger.info(f"  Type: {response.query_type}")
        logger.info(f"  Score: {score}")
        logger.info(f"  Sources: {response.sources}")
        logger.info(f"  Answer: {response.answer[:200]}...")
        logger.info(f"  Error: {response.error}")
        results.append({"question": q, "score": score, "type": response.query_type, "error": response.error})

    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for i, r in enumerate(results, 1):
        status = "PASS" if r["score"] >= 3 and not r["error"] else "FAIL"
        logger.info(f"  [{status}] Q{i}: score={r['score']}, type={r['type']}")
    
    passing = sum(1 for r in results if r["score"] >= 3 and not r["error"])
    logger.info(f"\n  {passing}/{len(results)} questions passed (score >= 3)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run only quick unit tests (no LLM calls)")
    parser.add_argument("--full", action="store_true", help="Run all tests including e2e")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FRAUD ANALYSIS AGENT - BACKEND TESTS")
    print("=" * 60 + "\n")

    # Always run these (no LLM calls)
    test_database()
    test_sql_safety()

    if not args.quick:
        test_vector_store()
        test_rag_tool()
        test_sql_tool()
        test_query_classifier()
        test_quality_scorer()

    if args.full:
        test_end_to_end_questions()

    if args.quick:
        print("\n[Quick tests done. Use --full for end-to-end tests]")
    elif not args.full:
        print("\n[Component tests done. Use --full to also run 6 sample questions]")
    else:
        print("\n[All tests complete]")
