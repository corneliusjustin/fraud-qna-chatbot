import os
import re
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_together_api_key() -> str:
    key = get_env("TOGETHER_API_KEY")
    if not key:
        raise ValueError("TOGETHER_API_KEY not set in .env")
    return key


def get_primary_model() -> str:
    return get_env("PRIMARY_MODEL", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")


def get_routing_model() -> str:
    return get_env("ROUTING_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")


def get_embedding_model() -> str:
    return get_env("EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-8k-retrieval")


def sanitize_input(text: str) -> str:
    text = text.strip()
    if len(text) > 2000:
        text = text[:2000]
    return text


def is_safe_sql(sql: str) -> bool:
    forbidden = re.compile(
        r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|PRAGMA|ATTACH|DETACH|REPLACE|TRUNCATE)\b',
        re.IGNORECASE
    )
    return not forbidden.search(sql)


def format_sql_result_as_text(columns: list[str], rows: list[list], max_rows: int = 20) -> str:
    if not rows:
        return "No results found."

    display_rows = rows[:max_rows]
    col_widths = [len(str(c)) for c in columns]
    for row in display_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header = " | ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    lines = [header, separator]
    for row in display_rows:
        line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        lines.append(line)

    if len(rows) > max_rows:
        lines.append(f"... and {len(rows) - max_rows} more rows")

    return "\n".join(lines)
