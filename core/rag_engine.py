"""
RAG Engine — Retrieval-Augmented Generation core pipeline.

This module provides the :class:`RAGEngine` class, which encapsulates the
three main stages of a RAG workflow:

1. **Embed** — Convert raw text chunks into dense vector representations.
2. **Retrieve** — Rank stored chunks against a user query via cosine similarity.
3. **Generate** — Build a grounded prompt from retrieved context and call an LLM.

It also exposes helpers for debug-printing retrieval results and producing an
OpenAI-compatible tool schema that lets an agent call the vector database
programmatically.

Typical usage::

    chunks     = RecursiveChunk.chunk(document_text)
    embeddings = RAGEngine.embed_chunks(chunks)

    results  = RAGEngine.retrieve(question, chunks, embeddings)
    context  = RAGEngine.build_context(results)
    answer   = RAGEngine.generate_answer(question, context)
"""

import math
from typing import Optional

from core.embeddings import OpenRouterEmbeddingClient
from core.models import RetrievalResult
from services.agent_completion import agent_complete
from config import settings


class RAGEngine:
    """Stateless utility class implementing a Retrieval-Augmented Generation pipeline.

    All methods are class-methods so the engine can be used without
    instantiation. The class constant :attr:`TOP_K` is read from application
    settings but can be overridden per call.

    Attributes:
        TOP_K (int): Default number of top-ranked chunks to return during
            retrieval. Sourced from ``settings.SEARCH_K``.
    """

    TOP_K: int = settings.SEARCH_K

    # ------------------------------------------------------------------ #
    #  Embedding                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def embed_chunks(cls, chunks: list[str]) -> list[list[float]]:
        """Embed a list of text chunks into dense vector representations.

        Delegates to :class:`~core.embeddings.OpenRouterEmbeddingClient` to
        produce one embedding vector per chunk. These vectors are later used
        for similarity search in :meth:`retrieve`.

        Args:
            chunks: A list of plain-text strings, typically produced by the
                chunking pipeline.

        Returns:
            A list of float vectors where ``embeddings[i]`` corresponds to
            ``chunks[i]``. Dimensionality is determined by the configured
            embedding model.

        Example::

            chunks = ["The sky is blue.", "Water boils at 100 °C."]
            embeddings = RAGEngine.embed_chunks(chunks)
            # embeddings -> [[0.12, -0.34, ...], [0.56, 0.78, ...]]
        """
        embeddings = OpenRouterEmbeddingClient.embed_documents(chunks)
        return embeddings

    # ------------------------------------------------------------------ #
    #  Similarity                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def cosine_similarity(
        cls,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Compute the cosine similarity between two vectors.

        Cosine similarity measures the angle between two vectors in a
        high-dimensional space, returning a value in ``[-1, 1]`` where
        ``1`` means identical direction (maximum relevance) and ``0`` means
        orthogonal (no relation).

        The formula is:

        .. math::

            \\text{cosine\\_sim}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\cdot \\|B\\|}

        Args:
            vector_a: First embedding vector.
            vector_b: Second embedding vector.

        Returns:
            A float in ``[-1.0, 1.0]`` representing the cosine similarity.
            Returns ``0.0`` if either vector is the zero vector to avoid
            division by zero.

        Note:
            Both vectors must have the same dimensionality. Mismatched lengths
            will silently truncate via :func:`zip`.
        """
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        length_a = math.sqrt(sum(a * a for a in vector_a))
        length_b = math.sqrt(sum(b * b for b in vector_b))

        # Guard against zero-vector inputs to prevent division by zero.
        if length_a == 0 or length_b == 0:
            return 0.0

        return dot_product / (length_a * length_b)

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def retrieve(
        cls,
        question: str,
        chunks: list[str],
        embeddings: list[list[float]],
        top_k: int = TOP_K,
    ) -> list[RetrievalResult]:
        """Retrieve the most semantically relevant chunks for a given question.

        Embeds the question, computes cosine similarity against every stored
        chunk embedding, sorts results in descending order of relevance, and
        returns the top-*k* matches.

        Args:
            question: The user's natural-language query.
            chunks: The full list of text chunks from the ingested document.
            embeddings: Pre-computed embedding vectors corresponding to
                ``chunks`` (produced by :meth:`embed_chunks`).
            top_k: Maximum number of results to return. Defaults to
                :attr:`TOP_K`.

        Returns:
            A list of :class:`~core.models.RetrievalResult` objects sorted by
            descending similarity score, containing at most ``top_k`` entries.

        Raises:
            IndexError: If ``embeddings`` is shorter than ``chunks``.
        """
        # Embed the query using the same model used at ingest time.
        question_embedding = OpenRouterEmbeddingClient.embed_query(question)
        results: list[RetrievalResult] = []

        for i, chunk in enumerate(chunks):
            score = cls.cosine_similarity(question_embedding, embeddings[i])
            results.append(
                RetrievalResult(
                    score=score,
                    chunk_id=i,
                    chunk_text=chunk,
                )
            )

        # Sort descending so the most relevant chunk is first.
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------ #
    #  Context building                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_context(cls, results: list[RetrievalResult]) -> str:
        """Assemble a labelled context string from retrieval results.

        Each result is formatted as a labelled block so the LLM can cite
        its source (e.g. ``[Page 2, Chunk 3]``). Blocks are separated by
        a blank line for readability.

        Args:
            results: Ordered list of retrieval results, typically the output
                of :meth:`retrieve`.

        Returns:
            A single multi-line string ready to be injected into an LLM
            prompt. Returns an empty string when ``results`` is empty.

        Example output::

            [Page 1, Chunk 1]
            The company reported revenue of $4.2 billion in 2023…

            [Page 1, Chunk 3]
            Operating expenses increased by 12 % year-on-year…
        """
        context_parts: list[str] = []
        for result in results:
            label = f"[{result.label()}]"
            context_parts.append(f"{label}\n{result.chunk_text}")

        return "\n\n".join(context_parts)

    # ------------------------------------------------------------------ #
    #  Answer generation                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def generate_answer(cls, question: str, context: str) -> str:
        """Generate a grounded answer from retrieved context using an LLM.

        Constructs a strict system prompt that instructs the model to answer
        *only* from the provided context, then calls the configured LLM
        completion endpoint. If the context is empty or whitespace-only, a
        fallback message is returned immediately without making an API call.

        Args:
            question: The user's original question.
            context: The context string produced by :meth:`build_context`.

        Returns:
            A string containing the LLM's answer. The model is instructed to
            cite the chunk label (e.g. ``[Page 1, Chunk 2]``) it used and to
            keep the answer concise.
        """
        if not context.strip():
            return "I could not find enough context to answer that."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict PDF question-answering assistant. "
                    "Answer using only the provided context. "
                    "Do not use outside knowledge. "
                    "If the context does not contain the answer, make a reasonable inference "
                    "from the context and clearly label it as inferred. "
                    "Mention the chunk label you used, like [Page 1, Chunk 2]. "
                    "Keep the answer short."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer only from the context above."
                ),
            },
        ]

        completion = agent_complete(messages)
        answer: str = completion.choices[0].message.content
        return answer

    # ------------------------------------------------------------------ #
    #  Debug utilities                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def print_retrieval_debug(
        cls,
        top_results: list[RetrievalResult],
        MIN_SCORE: Optional[float] = None,
    ) -> None:
        """Print a formatted debug table of retrieval results to stdout.

        Useful during development to inspect which chunks were retrieved and
        whether they exceed a quality threshold. When ``MIN_SCORE`` is given,
        each result is labelled ✓ PASS or ✗ FAIL.

        Args:
            top_results: The list of :class:`~core.models.RetrievalResult`
                objects to display.
            MIN_SCORE: Optional score threshold. Results at or above this
                value are marked as passing. When ``None``, no threshold
                labels are shown.

        Returns:
            ``None``. Output is written directly to stdout.

        Example output (with threshold)::

            ── Retrieval Debug (threshold: 0.75) ──────────────────────
              [✓ PASS] Page 2, Chunk 1, vector, Score: 0.8921
                       The company reported revenue of $4.2 billion…

              [✗ FAIL] Page 4, Chunk 3, keyword, Score: 0.6134
                       Operating expenses increased by 12 %…
            ────────────────────────────────────────────────────────────
        """
        threshold_label = f" (threshold: {MIN_SCORE})" if MIN_SCORE is not None else ""
        print(f"\n── Retrieval Debug{threshold_label} ──────────────────────")

        for result in top_results:
            if MIN_SCORE is not None:
                status = "✓ PASS" if result.score >= MIN_SCORE else "✗ FAIL"
                print(
                    f"  [{status}] {result.label()}, "
                    f"{result.retrieval_method}, Score: {result.score:.4f}"
                )
            else:
                print(
                    f"  {result.label()}, "
                    f"{result.retrieval_method}, Score: {result.score:.4f}"
                )

            snippet = result.chunk_text[:120].replace("\n", " ")
            print(f"           {snippet}…")
            print()

        print("────────────────────────────────────────────────────")

    # ------------------------------------------------------------------ #
    #  Tool schema                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_tools(cls) -> list[dict]:
        """Return the OpenAI-compatible tool schema for the PDF search function.

        This schema is passed to the LLM so it can decide when to invoke the
        vector database search tool. The schema conforms to the ``tools``
        parameter format used by the OpenAI Chat Completions API.

        Returns:
            A list containing a single tool-definition dict with keys
            ``"type"`` and ``"function"``. The function expects a ``query``
            string argument.

        Example::

            tools = RAGEngine.get_tools()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_pdf_database",
                    "description": (
                        "Search the Neon vector database for facts inside a specific PDF document. "
                        "Use this whenever the user asks about information contained in their uploaded files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The specific search query to look for in the database "
                                    "(e.g., 'financial revenue 2023')."
                                ),
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
    @classmethod
    def keyword_overlap_score(
        cls,
        query: str,
        text: str
    ) -> float:

        query_words = {
            word.lower().strip(".,:;()[]{}")
            for word in query.split()
            if len(word) >2
        }
        text_words = {
            word.lower().strip(".,:;()[]{}")
            for word in text.split()
        }
        if not query_words:
            return 0.0

        overlap = query_words.intersection(text_words)

        return len(overlap) / len(query_words)
    
    @classmethod
    def rerank_results(
        cls,
        query: str,
        results: list[RetrievalResult],
        keyword_weight: float = 0.05
    ) -> list[RetrievalResult]:
        
        reranked = []

        for result in results:
            overlap_score = cls.keyword_overlap_score(
                query=query,
                text=result.chunk_text,
            )
            result.score = result.score + (keyword_weight * overlap_score)
            result.retrieval_method = f"{result.retrieval_method}+rerank"
            reranked.append(result)
        
        reranked.sort(key=lambda result: result.score, reverse=True)

        return reranked
