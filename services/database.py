import sqlite3
import logging
import os
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "fraud_database.db"
DATASET_DIR = Path(__file__).parent.parent / "dataset"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.Connection(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def is_database_ready() -> bool:
    if not DB_PATH.exists():
        return False
    try:
        conn = get_connection()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fraud_transactions'"
        )
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    except Exception:
        return False


def get_table_schema() -> str:
    return """Table: fraud_transactions
Columns:
  - row_index (INTEGER): Original row index
  - trans_date_trans_time (TEXT): Transaction datetime as 'YYYY-MM-DD HH:MM:SS'
  - cc_num (INTEGER): Credit card number
  - merchant (TEXT): Merchant name (prefixed with 'fraud_')
  - category (TEXT): Transaction category (e.g., 'misc_net', 'grocery_pos', 'shopping_net')
  - amt (REAL): Transaction amount in USD
  - first (TEXT): Cardholder first name
  - last (TEXT): Cardholder last name
  - gender (TEXT): Cardholder gender ('M' or 'F')
  - street (TEXT): Cardholder street address
  - city (TEXT): Cardholder city
  - state (TEXT): Cardholder state (2-letter code)
  - zip (INTEGER): Cardholder ZIP code
  - lat (REAL): Cardholder latitude
  - long (REAL): Cardholder longitude
  - city_pop (INTEGER): City population
  - job (TEXT): Cardholder job title
  - dob (TEXT): Date of birth as 'YYYY-MM-DD'
  - trans_num (TEXT): Unique transaction ID
  - unix_time (INTEGER): Unix timestamp
  - merch_lat (REAL): Merchant latitude
  - merch_long (REAL): Merchant longitude
  - is_fraud (INTEGER): 1 = fraudulent, 0 = legitimate

Date range: 2019-01-01 to 2020-12-31
Use strftime() for date grouping. Example: strftime('%Y-%m', trans_date_trans_time)
"""


def setup_database() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()

    train_path = DATASET_DIR / "fraudTrain.csv"
    test_path = DATASET_DIR / "fraudTest.csv"

    stats = {"train_rows": 0, "test_rows": 0, "total_rows": 0, "fraud_count": 0}

    logger.info("Loading training data...")
    df_train = pd.read_csv(str(train_path))
    df_train.rename(columns={"Unnamed: 0": "row_index"}, inplace=True)
    if "" in df_train.columns:
        df_train.rename(columns={"": "row_index"}, inplace=True)
    stats["train_rows"] = len(df_train)

    logger.info("Loading test data...")
    df_test = pd.read_csv(str(test_path))
    df_test.rename(columns={"Unnamed: 0": "row_index"}, inplace=True)
    if "" in df_test.columns:
        df_test.rename(columns={"": "row_index"}, inplace=True)
    stats["test_rows"] = len(df_test)

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Normalize datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["dob"] = pd.to_datetime(df["dob"]).dt.strftime("%Y-%m-%d")

    logger.info(f"Writing {len(df)} rows to SQLite...")
    df.to_sql("fraud_transactions", conn, if_exists="replace", index=False)

    # Create indexes for performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_date ON fraud_transactions(trans_date_trans_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_is_fraud ON fraud_transactions(is_fraud)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON fraud_transactions(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_merchant ON fraud_transactions(merchant)")
    conn.commit()

    stats["total_rows"] = len(df)
    cursor = conn.execute("SELECT COUNT(*) FROM fraud_transactions WHERE is_fraud = 1")
    stats["fraud_count"] = cursor.fetchone()[0]

    conn.close()
    logger.info(f"Database setup complete: {stats}")
    return stats


def execute_query(sql: str, timeout: int = 10) -> tuple[list[str], list[list]]:
    conn = get_connection()
    try:
        conn.execute(f"PRAGMA busy_timeout={timeout * 1000}")
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchall()]
        return columns, rows
    finally:
        conn.close()


def validate_database() -> dict:
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM fraud_transactions")
        total = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM fraud_transactions WHERE is_fraud = 1")
        fraud = cursor.fetchone()[0]

        cursor = conn.execute("SELECT MIN(trans_date_trans_time), MAX(trans_date_trans_time) FROM fraud_transactions")
        date_range = cursor.fetchone()

        cursor = conn.execute("SELECT DISTINCT category FROM fraud_transactions ORDER BY category")
        categories = [row[0] for row in cursor.fetchall()]

        return {
            "total_rows": total,
            "fraud_count": fraud,
            "legitimate_count": total - fraud,
            "fraud_rate": round(fraud / total * 100, 2) if total > 0 else 0,
            "date_range": {"min": date_range[0], "max": date_range[1]},
            "categories": categories,
        }
    finally:
        conn.close()
