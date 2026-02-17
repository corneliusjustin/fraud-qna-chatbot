import re
import sqlite3
import logging
from services.database import execute_query, get_connection, get_table_schema
from services.together_ai import chat_completion_routing
from models.schemas import SQLResult
from utils.helpers import is_safe_sql

logger = logging.getLogger(__name__)

SQL_SYSTEM_PROMPT = """You are an expert SQL query generator for SQLite databases.
You will be given a natural language question and must generate a valid SQLite SELECT query.

{schema}

RULES:
1. ONLY generate SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA.
2. Use strftime() for date grouping. Examples:
   - Monthly: strftime('%Y-%m', trans_date_trans_time)
   - Daily: strftime('%Y-%m-%d', trans_date_trans_time)
   - Yearly: strftime('%Y', trans_date_trans_time)
3. Always include appropriate WHERE clauses when filtering.
4. Use LIMIT 100 for non-aggregation queries.
5. Use ROUND() for decimal values.
6. For fraud rate calculations: ROUND(AVG(is_fraud) * 100, 2) or ROUND(CAST(SUM(is_fraud) AS REAL) / COUNT(*) * 100, 2)

EXAMPLES:

Question: "What is the monthly fraud rate?"
SQL: SELECT strftime('%Y-%m', trans_date_trans_time) AS month, ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct FROM fraud_transactions GROUP BY month ORDER BY month

Question: "Which categories have the most fraud?"
SQL: SELECT category, COUNT(*) AS fraud_count, ROUND(CAST(COUNT(*) AS REAL) / (SELECT COUNT(*) FROM fraud_transactions WHERE is_fraud = 1) * 100, 2) AS pct_of_total_fraud FROM fraud_transactions WHERE is_fraud = 1 GROUP BY category ORDER BY fraud_count DESC LIMIT 10

Question: "What is the average fraudulent transaction amount?"
SQL: SELECT ROUND(AVG(amt), 2) AS avg_fraud_amount FROM fraud_transactions WHERE is_fraud = 1

OUTPUT FORMAT:
Return ONLY the SQL query, nothing else. No markdown, no explanation, no code blocks.
"""


def generate_sql(question: str, model: str | None = None) -> str:
    schema = get_table_schema()
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": f"Generate a SQLite SELECT query for this question:\n\n{question}"},
    ]
    
    raw = chat_completion_routing(messages)
    
    # Clean up the response - strip markdown code blocks if present
    sql = raw.strip()
    sql = re.sub(r'^```(?:sql)?\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    sql = sql.strip().rstrip(';') + ';' if not sql.strip().endswith(';') else sql.strip()
    
    return sql


def validate_sql(sql: str) -> tuple[bool, str]:
    # Check safety
    if not is_safe_sql(sql):
        return False, "Query contains forbidden operations (only SELECT is allowed)"
    
    # Check it starts with SELECT
    if not sql.strip().upper().startswith("SELECT"):
        return False, "Query must start with SELECT"
    
    # Try EXPLAIN to check syntax
    try:
        conn = get_connection()
        conn.execute(f"EXPLAIN {sql.rstrip(';')}")
        conn.close()
        return True, "Valid"
    except sqlite3.OperationalError as e:
        return False, f"SQL syntax error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def run_sql_query(question: str, max_retries: int = 2) -> SQLResult:
    last_error = ""
    
    for attempt in range(max_retries + 1):
        try:
            # Generate SQL
            sql = generate_sql(question)
            logger.info(f"Generated SQL (attempt {attempt + 1}): {sql}")
            
            # Validate
            is_valid, msg = validate_sql(sql)
            if not is_valid:
                last_error = msg
                logger.warning(f"SQL validation failed: {msg}")
                if attempt < max_retries:
                    continue
                return SQLResult(query=sql, error=f"Invalid SQL after {max_retries + 1} attempts: {msg}")
            
            # Execute
            columns, rows = execute_query(sql)
            
            # Limit rows
            if len(rows) > 100:
                rows = rows[:100]
            
            return SQLResult(
                query=sql,
                columns=columns,
                rows=rows,
                row_count=len(rows),
            )
            
        except sqlite3.OperationalError as e:
            last_error = str(e)
            logger.warning(f"SQL execution error (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
                
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error in SQL tool: {e}")
            break
    
    return SQLResult(query="", error=f"SQL query failed after {max_retries + 1} attempts: {last_error}")
