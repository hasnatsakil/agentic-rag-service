"""
Text chunking utilities for the RAG ingestion pipeline.

This module provides :class:`RecursiveChunk`, a hierarchical text splitter
that progressively breaks documents into smaller pieces:

1. **Paragraphs** — split on double newlines (``\\n\\n``).
2. **Sentences**  — split on full stops (``.``).
3. **Words**      — sliding window with configurable size and overlap.

The strategy ensures that chunks never exceed ``max_words`` while preserving
as much natural linguistic structure as possible.
"""


class RecursiveChunk:
    """Hierarchical text splitter with paragraph → sentence → word fallback.

    Splits text at progressively finer granularities until every resulting
    piece fits within the configured word limit.  An optional word overlap
    between adjacent word-level chunks helps preserve context across
    boundaries.

    Class Attributes:
        max_words (int): Default maximum number of words per chunk (``40``).
        overlap (int): Default number of overlapping words between consecutive
            word-level chunks (``5``).
    """

    max_words: int = 150
    overlap: int = 25

    # ------------------------------------------------------------------ #
    #  Configuration helpers                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_config(
        cls,
        max_words: int | None = None,
        overlap: int | None = None,
    ) -> tuple[int, int]:
        """Resolve per-call overrides against class-level defaults.

        Args:
            max_words: Override for :attr:`max_words`. Uses the class default
                when ``None``.
            overlap: Override for :attr:`overlap`. Uses the class default
                when ``None``.

        Returns:
            A ``(max_words, overlap)`` tuple with all ``None`` values
            replaced by their class-level defaults.
        """
        if max_words is None:
            max_words = cls.max_words
        if overlap is None:
            overlap = cls.overlap
        return max_words, overlap

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean(parts: list[str]) -> list[str]:
        """Strip and remove empty strings from any list of text parts.

        Args:
            parts: Raw text segments, potentially containing blank strings
                or leading/trailing whitespace.

        Returns:
            A new list with each part stripped and empty strings discarded.
        """
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _fits(text: str, max_words: int) -> bool:
        """Check if text is within the word limit.

        Args:
            text: The text string to evaluate.
            max_words: The maximum number of words allowed.

        Returns:
            ``True`` if the word count of ``text`` is at or below
            ``max_words``, ``False`` otherwise.
        """
        return len(text.split()) <= max_words

    # ------------------------------------------------------------------ #
    #  Splitting strategies                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def chunk_by_paragraph(cls, text: str) -> list[str]:
        """Split text into paragraphs on double newlines.

        Args:
            text: The raw input text.

        Returns:
            A cleaned list of paragraph strings with blank entries removed.
        """
        return cls._clean(text.split("\n\n"))

    @classmethod
    def chunk_by_sentence(cls, text: str) -> list[str]:
        """Split text into sentences on full stops.

        Each resulting sentence has its terminating period re-appended to
        preserve grammatical structure.

        Args:
            text: A paragraph or block of text to sentence-tokenise.

        Returns:
            A cleaned list of sentence strings, each ending with ``"."``.
        """
        return [s + "." for s in cls._clean(text.split("."))]

    @staticmethod
    def chunk_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into fixed-size word windows with overlap.

        Uses a sliding window approach: each window starts ``chunk_size -
        overlap`` words after the previous one, so adjacent chunks share
        ``overlap`` words at their boundaries.

        Args:
            text: The text to split.
            chunk_size: Maximum number of words per chunk.
            overlap: Number of words shared between consecutive chunks.

        Returns:
            A list of space-joined word-window strings.

        Example::

            chunks = RecursiveChunk.chunk_by_words("a b c d e", 3, 1)
            # -> ["a b c", "c d e"]
        """
        words = text.split()
        step = chunk_size - overlap
        return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), step)]

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def chunk(
        cls,
        text: str,
        max_words: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Recursively split text into chunks that fit within ``max_words``.

        Applies splitting strategies in order of decreasing granularity:

        1. Splits into paragraphs. Paragraphs that fit are kept as-is.
        2. Oversized paragraphs are split into sentences. Sentences that fit
           are kept as-is.
        3. Oversized sentences are split into word windows via
           :meth:`chunk_by_words`.

        Args:
            text: The raw input text to chunk.
            max_words: Maximum number of words per chunk. Defaults to
                :attr:`max_words` when ``None``.
            overlap: Number of overlapping words between word-level chunks.
                Defaults to :attr:`overlap` when ``None``.

        Returns:
            A flat list of text chunk strings, each containing at most
            ``max_words`` words.
        """
        max_words, overlap = cls._resolve_config(max_words, overlap)
        final_chunks: list[str] = []

        for paragraph in cls.chunk_by_paragraph(text):
            # Paragraph fits — keep it whole.
            if cls._fits(paragraph, max_words):
                final_chunks.append(paragraph)
                continue

            # Paragraph too large — try sentence-level splitting.
            for sentence in cls.chunk_by_sentence(paragraph):
                if cls._fits(sentence, max_words):
                    final_chunks.append(sentence)
                else:
                    # Sentence still too large — fall back to word windows.
                    final_chunks.extend(
                        cls.chunk_by_words(
                            sentence,
                            chunk_size=max_words,
                            overlap=overlap,
                        )
                    )

        return final_chunks
