from enum import Enum


class QueryType(str, Enum):
    SQL = "sql"
    RAG = "rag"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ErrorType(str, Enum):
    LLM_ERROR = "llm_error"
    SQL_ERROR = "sql_error"
    RAG_ERROR = "rag_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"
