import logging
from models.enums import ErrorType

logger = logging.getLogger(__name__)


class AgentError(Exception):
    def __init__(self, message: str, error_type: ErrorType, details: str = ""):
        self.message = message
        self.error_type = error_type
        self.details = details
        super().__init__(self.message)


def handle_llm_error(e: Exception) -> str:
    logger.error(f"LLM error: {e}")
    if "rate_limit" in str(e).lower() or "429" in str(e):
        return "The AI service is temporarily rate-limited. Please try again in a moment."
    if "timeout" in str(e).lower():
        return "The AI service timed out. Please try again."
    if "500" in str(e) or "502" in str(e) or "503" in str(e):
        return "The AI service is temporarily unavailable. Please try again later."
    return f"An error occurred with the AI service: {str(e)}"


def handle_sql_error(e: Exception) -> str:
    logger.error(f"SQL error: {e}")
    if "syntax" in str(e).lower():
        return "There was a syntax error in the generated SQL query. Retrying..."
    if "no such table" in str(e).lower():
        return "The database table was not found. Please run data setup first."
    if "no such column" in str(e).lower():
        return "The query referenced a column that doesn't exist in the database."
    return f"A database error occurred: {str(e)}"


def handle_rag_error(e: Exception) -> str:
    logger.error(f"RAG error: {e}")
    return f"An error occurred during document retrieval: {str(e)}"


def safe_execute(func, *args, error_type: ErrorType = ErrorType.UNKNOWN_ERROR, **kwargs):
    try:
        return func(*args, **kwargs)
    except AgentError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        raise AgentError(
            message=str(e),
            error_type=error_type,
            details=f"Function: {func.__name__}"
        )
