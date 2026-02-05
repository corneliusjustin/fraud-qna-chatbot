import logging
from collections.abc import Generator
from openai import OpenAI
from utils.helpers import get_together_api_key, get_primary_model, get_routing_model, get_embedding_model

logger = logging.getLogger(__name__)

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_together_api_key(),
            base_url="https://api.together.xyz/v1",
        )
    return _client


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    client = get_client()
    model = model or get_primary_model()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def chat_completion_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> Generator[str, None, None]:
    client = get_client()
    model = model or get_primary_model()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def chat_completion_routing(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    return chat_completion(
        messages=messages,
        model=get_routing_model(),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embeddings(texts: list[str], max_chars_per_text: int = 400) -> list[list[float]]:
    client = get_client()
    model = get_embedding_model()
    # Truncate texts to stay within token limits
    truncated = [t[:max_chars_per_text] for t in texts]
    all_embeddings = []
    # Send each text individually to avoid exceeding total token limit
    for t in truncated:
        response = client.embeddings.create(
            model=model,
            input=[t],
        )
        all_embeddings.append(response.data[0].embedding)
    return all_embeddings
