from pydantic import BaseModel, Field
from typing import Optional
from models.enums import QueryType


class ClassificationResult(BaseModel):
    query_type: QueryType = Field(description="Type of query: sql, rag, or hybrid")
    reasoning: str = Field(description="Brief reasoning for the classification")
    sql_query_hint: Optional[str] = Field(default=None, description="Hint for SQL query if applicable")
    rag_search_hint: Optional[str] = Field(default=None, description="Hint for RAG search if applicable")


class SQLResult(BaseModel):
    query: str = Field(description="The SQL query that was executed")
    columns: list[str] = Field(default_factory=list, description="Column names")
    rows: list[list] = Field(default_factory=list, description="Result rows")
    row_count: int = Field(default=0, description="Number of rows returned")
    error: Optional[str] = Field(default=None, description="Error message if query failed")


class RAGResult(BaseModel):
    chunks: list[str] = Field(default_factory=list, description="Retrieved text chunks")
    metadatas: list[dict] = Field(default_factory=list, description="Metadata for each chunk")
    distances: list[float] = Field(default_factory=list, description="Distance scores")
    error: Optional[str] = Field(default=None, description="Error message if retrieval failed")


class QualityScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Quality score from 1 to 5")
    reasoning: str = Field(description="Reasoning for the score")
    has_hallucination: bool = Field(default=False, description="Whether hallucination was detected")
    missing_information: list[str] = Field(default_factory=list, description="List of missing information")


class AgentResponse(BaseModel):
    answer: str = Field(description="The final answer text")
    query_type: QueryType = Field(description="Type of query that was processed")
    sql_result: Optional[SQLResult] = Field(default=None, description="SQL result if applicable")
    rag_result: Optional[RAGResult] = Field(default=None, description="RAG result if applicable")
    quality_score: Optional[QualityScore] = Field(default=None, description="Quality score")
    sources: list[str] = Field(default_factory=list, description="Source citations")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")
    retry_count: int = Field(default=0, description="Number of retries performed")
