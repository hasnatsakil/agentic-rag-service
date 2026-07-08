"""
Embedding client for the OpenRouter API.

This module wraps the OpenRouter embedding endpoint, exposing a high-level
interface for converting text into dense vector representations.  It supports
automatic model fallback: if the primary embedding model fails (e.g. rate
limit, downtime), the client transparently retries with the next model in the
priority list defined by :attr:`~config.openrouter_settings.RouterConfig.EMBEDDING_MODELS_PRIORITY`.
"""

from __future__ import annotations

import structlog
from config.openrouter_settings import client, RouterConfig
from openai.types.create_embedding_response import CreateEmbeddingResponse

logger = structlog.get_logger(__name__)

# Commented-out batch/truncation limits kept for future reference:
# _BATCH_LIMIT = 15        # OpenRouter embedding API limit per request
# _MAX_CHARS   = 5000      # Max characters per text (~1200 tokens)


class OpenRouterEmbeddingClient:
    """Client for generating text embeddings via the OpenRouter API.

    All methods are class-methods; no instantiation is required.  The client
    iterates through :attr:`~config.openrouter_settings.RouterConfig.EMBEDDING_MODELS_PRIORITY`
    and uses the first model that succeeds, providing silent resilience against
    individual model failures.
    """

    @classmethod
    def _execute_embedding(
        cls,
        input_data: str | list[str],
    ) -> CreateEmbeddingResponse:
        """Call the OpenRouter embeddings API with automatic model fallback.

        Iterates through the configured embedding model priority list and
        returns the first successful response.  Logs a warning for each
        model failure before trying the next one.

        Args:
            input_data: Either a single string (for query embedding) or a
                list of strings (for batch document embedding).

        Returns:
            A :class:`~openai.types.create_embedding_response.CreateEmbeddingResponse`
            containing one embedding object per input string.

        Raises:
            Exception: If every model in the priority list fails.
        """
        for model in RouterConfig.EMBEDDING_MODELS_PRIORITY:
            try:
                return client.embeddings.create(
                    model=model,
                    input=input_data,
                )
            except Exception as e:
                logger.warning(f"Embedding failed for {model}. Error: {str(e)}")
                continue

        raise Exception("All embedding models failed!")

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        """Embed a single query string into a dense vector.

        Intended for embedding user questions at query time.  The resulting
        vector is compared against pre-computed document chunk embeddings
        stored in the vector store.

        Args:
            text: The user query or search string to embed.

        Returns:
            A list of floats representing the embedding vector. Dimensionality
            is determined by the active embedding model (e.g. 1536 for
            ``text-embedding-3-small``).
        """
        response = cls._execute_embedding(text)
        return response.data[0].embedding

    @classmethod
    def embed_documents(cls, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks into dense vectors.

        Intended for embedding text chunks at ingest time. The returned
        vectors are stored alongside the chunks in the Neon vector store for
        later similarity search.

        Args:
            texts: A list of text strings to embed in a single API call.

        Returns:
            A list of embedding vectors where ``result[i]`` corresponds to
            ``texts[i]``. Each vector is a list of floats with dimensionality
            determined by the active model.
        """
        response = cls._execute_embedding(texts)
        return [item.embedding for item in response.data]
