class RecursiveChunk:
    max_words: int = 40
    overlap: int = 5

    @classmethod
    def _resolve_config(
        cls, max_words=None, 
        overlap=None
        ):
        if max_words is None:
            max_words = cls.max_words

        if overlap is None:
            overlap = cls.overlap

        return max_words, overlap

    @staticmethod
    def _clean(
            parts: list[str]
            )-> list[str]:
        """Strip and remove empty strings from any list of text parts."""
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _fits(
            text: str, 
            max_words: int
            ) -> bool:
        """Check if text is within the word limit."""
        return len(text.split()) <= max_words

    @classmethod
    def chunk_by_paragraph(
            cls,
            text: str
            ) -> list[str]:
        return cls._clean(text.split('\n\n'))

    @classmethod
    def chunk_by_sentence(
            cls,
            text: str
            ) -> list[str]:
        return [s + "." for s in cls._clean(text.split("."))]

    @staticmethod
    def chunk_by_words(
            text: str, 
            chunk_size: int, 
            overlap: int
            ) -> list[str]:
        words = text.split()
        step = chunk_size - overlap
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]
    
    @classmethod
    def chunk(
        cls, 
        text, 
        max_words=None, 
        overlap=None):
        max_words, overlap = cls._resolve_config(max_words, overlap)

        final_chunks = []

        for paragraph in cls.chunk_by_paragraph(text):
            if cls._fits(paragraph, max_words):
                final_chunks.append(paragraph)
                continue

            for sentence in cls.chunk_by_sentence(paragraph):
                if cls._fits(sentence, max_words):
                    final_chunks.append(sentence)
                else:
                    final_chunks.extend(
                        cls.chunk_by_words(
                            sentence,
                            chunk_size=max_words,
                            overlap=overlap,
                        )
                    )

        return final_chunks

