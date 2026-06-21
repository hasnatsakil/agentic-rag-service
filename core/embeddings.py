from __future__ import annotations

# import time
import structlog
from config.openrouter_settings import RouterEmbeddingModel, client

logger = structlog.get_logger(__name__)

# OpenRouter embedding API limit per request
# _BATCH_LIMIT = 15
# Max characters per text before truncation (roughly ~1200 tokens)
# _MAX_CHARS = 5000

class OpenRouterEmbeddingClient:
    MODEL: str = RouterEmbeddingModel.SMALL.value

    @classmethod
    def embed_query(
        cls, 
        text: str
        ) -> list[float]:
        response = client.embeddings.create(
            model=cls.MODEL,
            input=text,
        )
        return response.data[0].embedding

    @classmethod
    def embed_documents(
        cls, 
        texts: list[str]
        ) -> list[list[float]]:
        response = client.embeddings.create(
            model=cls.MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]
