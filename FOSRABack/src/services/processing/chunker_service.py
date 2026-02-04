from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from FOSRABack.src.api.request_context import RequestContext

from FOSRABack.src.domain.exceptions import ChunkingError
from FOSRABack.src.storage.utils.converters import ulid_factory
from FOSRABack.src.domain.schemas import (
    Chunk,
    ChunkFull,
    SourceFull,
    ChunkerConfig,
    ChunkingResult,
)


from FOSRABack.src.domain.enums import ChunkerType


# =============================================================================
# Base Chunker Interface
# =============================================================================
class BaseChunker(ABC):
    _registry: dict[str, type["BaseChunker"]] = {}

    def __init__(self):
        pass

    @classmethod
    def register(cls, chunker_type: ChunkerType):
        def decorator(chunker_cls: type["BaseChunker"]):
            cls._registry[chunker_type.value] = chunker_cls
            return chunker_cls

        return decorator

    @classmethod
    def get_chunker(cls, chunker_type: ChunkerType) -> "BaseChunker | None":
        chunker_cls = cls._registry.get(chunker_type.value)
        if chunker_cls:
            return chunker_cls()
        return None

    @classmethod
    def get_available_chunkers(cls) -> list[str]:
        return list(cls._registry.keys())

    chunker_type: ChunkerType
    display_name: str

    @abstractmethod
    async def chunk_text(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        config: ChunkerConfig | None = None,
    ) -> ChunkingResult:
        pass

    async def chunk_source(
        self,
        source: SourceFull,
        config: ChunkerConfig | None = None,
        ctx: "RequestContext | None" = None,
    ) -> SourceFull:
        if not source.content:
            logger.warning(f"Source {source.source_id} has no content to chunk")
            return source

        result = await self.chunk_text(
            text=source.content,
            source_id=source.source_id,
            source_hash=source.hash or "",
            config=config,
        )

        if result.errors:
            logger.error(f"Chunking errors for {source.source_id}: {result.errors}")

        source.chunks = result.chunks
        return source

    def _create_chunk(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        start_index: int,
        end_index: int,
        token_count: int,
    ) -> ChunkFull:
        return ChunkFull(
            chunk_id=ulid_factory(),
            text=text,
            start_index=start_index,
            end_index=end_index,
            token_count=token_count,
            source_id=source_id,
            source_hash=source_hash,
        )

    def _create_error_result(self, error: str) -> ChunkingResult:
        return ChunkingResult(
            chunks=[],
            chunker_used=self.chunker_type.value,
            errors=[error],
        )


# =============================================================================
# Concrete Chunker Implementations
# =============================================================================


@BaseChunker.register(ChunkerType.SEMANTIC)
class SemanticChunker(BaseChunker):
    chunker_type = ChunkerType.SEMANTIC
    display_name = "Semantic Chunker"

    def __init__(self):
        super().__init__()
        self._chonkie_chunker = None

    async def _initialize_chunker(self, config: ChunkerConfig) -> None:
        if self._chonkie_chunker is None:
            from chonkie.chunker import SemanticChunker as ChonkieSemanticChunker

            self._chonkie_chunker = ChonkieSemanticChunker(
                embedding_model=config.embedding_model,
                chunk_size=config.chunk_size,
                threshold=config.similarity_threshold,
                similarity_window=4,
                min_sentences_per_chunk=3,
            )

    async def chunk_text(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        config: ChunkerConfig,
    ) -> ChunkingResult:
        import time

        start_time = time.time()
        config = config
        try:
            await self._initialize_chunker(config)

            if self._chonkie_chunker is None:
                return self._create_error_result("Failed to initialize chunker")

            # Run chunking in thread pool
            chonkie_chunks = await asyncio.to_thread(self._chonkie_chunker.chunk, text)

            # Convert to our Chunk format
            chunks = []
            total_tokens = 0

            for c in chonkie_chunks:
                chunk = self._create_chunk(
                    text=c.text,
                    source_id=source_id,
                    source_hash=source_hash,
                    start_index=c.start_index,
                    end_index=c.end_index,
                    token_count=c.token_count,
                )

                chunks.append(chunk)
                total_tokens += c.token_count

            chunk_time = (time.time() - start_time) * 1000

            return ChunkingResult(
                chunks=chunks,
                chunker_used=self.chunker_type.value,
                chunk_time_ms=chunk_time,
                total_tokens=total_tokens,
            )

        except Exception as e:
            raise ChunkingError(
                source_id=source_id,
                chunker_type=config.preferred_chunker_type,
                reason=str(e),
                remediation="Check chunker configuration and input text.",
            ) from e


@BaseChunker.register(ChunkerType.TOKEN)
class TokenChunker(BaseChunker):
    chunker_type = ChunkerType.TOKEN
    display_name = "Token Chunker"

    async def chunk_text(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        config: ChunkerConfig,
    ) -> ChunkingResult:
        import time

        start_time = time.time()
        config = config or ChunkerConfig()

        try:
            from chonkie.chunker import TokenChunker as ChonkieTokenChunker

            chunker = ChonkieTokenChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            chonkie_chunks = await asyncio.to_thread(chunker.chunk, text)  # pyright: ignore

            chunks = []
            total_tokens = 0

            for c in chonkie_chunks:
                chunk = self._create_chunk(
                    text=c.text,
                    source_id=source_id,
                    source_hash=source_hash,
                    start_index=c.start_index,
                    end_index=c.end_index,
                    token_count=c.token_count,
                )
                chunks.append(chunk)
                total_tokens += c.token_count

            chunk_time = (time.time() - start_time) * 1000

            return ChunkingResult(
                chunks=chunks,
                chunker_used=self.chunker_type.value,
                chunk_time_ms=chunk_time,
                total_tokens=total_tokens,
            )
        except Exception as e:
            logger.error(f"Token chunking failed: {e}")
            raise ChunkingError(
                source_id=source_id,
                chunker_type=config.preferred_chunker_type,
                reason=str(e),
                remediation="Check chunker configuration and input text.",
            ) from e


@BaseChunker.register(ChunkerType.SENTENCE)
class SentenceChunker(BaseChunker):
    chunker_type = ChunkerType.SENTENCE
    display_name = "Sentence Chunker"

    async def chunk_text(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        config: ChunkerConfig,
    ) -> ChunkingResult:
        import time

        start_time = time.time()
        config = config or ChunkerConfig()

        try:
            from chonkie.chunker import SentenceChunker as ChonkieSentenceChunker

            chunker = ChonkieSentenceChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            chonkie_chunks = await asyncio.to_thread(chunker.chunk, text)  # pyright: ignore

            chunks = []
            total_tokens = 0

            for c in chonkie_chunks:
                chunk = self._create_chunk(
                    text=c.text,
                    source_id=source_id,
                    source_hash=source_hash,
                    start_index=c.start_index,
                    end_index=c.end_index,
                    token_count=c.token_count,
                )
                chunks.append(chunk)
                total_tokens += c.token_count

            chunk_time = (time.time() - start_time) * 1000

            return ChunkingResult(
                chunks=chunks,
                chunker_used=self.chunker_type.value,
                chunk_time_ms=chunk_time,
                total_tokens=total_tokens,
            )

        except Exception as e:
            logger.error(f"Sentence chunking failed: {e}")
            raise ChunkingError(
                source_id=source_id,
                chunker_type=config.preferred_chunker_type,
                reason=str(e),
                remediation="Check chunker configuration and input text.",
            ) from e


@BaseChunker.register(ChunkerType.FIXED)
class FixedSizeChunker(BaseChunker):
    chunker_type = ChunkerType.FIXED
    display_name = "Fixed Size Chunker"

    async def chunk_text(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        config: ChunkerConfig,
    ) -> ChunkingResult:
        import time

        start_time = time.time()
        config = config or ChunkerConfig()

        try:
            chars_per_token = 4
            chunk_chars = config.chunk_size * chars_per_token
            overlap_chars = config.chunk_overlap * chars_per_token

            chunks = []
            total_tokens = 0
            start = 0

            while start < len(text):
                end = min(start + chunk_chars, len(text))

                if end < len(text):
                    last_space = text.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space

                chunk_text = text[start:end].strip()
                if chunk_text:
                    token_count = max(1, len(chunk_text) // chars_per_token)

                    chunk = self._create_chunk(
                        text=chunk_text,
                        source_id=source_id,
                        source_hash=source_hash,
                        start_index=start,
                        end_index=end,
                        token_count=token_count,
                    )
                    chunks.append(chunk)
                    total_tokens += token_count

                start = end - overlap_chars
                if start >= len(text) - overlap_chars:
                    break

            chunk_time = (time.time() - start_time) * 1000

            return ChunkingResult(
                chunks=chunks,
                chunker_used=self.chunker_type.value,
                chunk_time_ms=chunk_time,
                total_tokens=total_tokens,
            )

        except Exception as e:
            logger.error(f"Fixed-size chunking failed: {e}")
            raise ChunkingError(
                source_id=source_id,
                chunker_type=config.preferred_chunker_type,
                reason=str(e),
                remediation="Check chunker configuration and input text.",
            ) from e


# =============================================================================
# Chunker Service (Main Interface)
# =============================================================================


class ChunkerService:
    def __init__(
        self,
        default_chunker: ChunkerType = ChunkerType.SEMANTIC,
    ):
        self.default_chunker = default_chunker

    def get_chunker(self, chunker_type: ChunkerType) -> BaseChunker | None:
        """Get a chunker instance by type."""
        return BaseChunker.get_chunker(chunker_type)

    def get_available_chunkers(self) -> list[str]:
        """Get list of all available chunker types."""
        return BaseChunker.get_available_chunkers()

    async def chunk_text(
        self,
        text: str,
        source_id: str,
        source_hash: str,
        config: ChunkerConfig | None = None,
    ) -> ChunkingResult:
        chunker_type = config.preferred_chunker_type or self.default_chunker
        chunker = self.get_chunker(chunker_type)

        if not chunker:
            return ChunkingResult(
                chunks=[],
                errors=[f"Chunker not found: {chunker_type}"],
            )

        return await chunker.chunk_text(text, source_id, source_hash, config)

    async def chunk_source(
        self,
        config: ChunkerConfig,
        source: SourceFull,
    ) -> SourceFull:
        chunker_type = config.preferred_chunker_type or self.default_chunker

        chunker = self.get_chunker(chunker_type)

        if not chunker:
            logger.error(f"Chunker not found: {chunker_type}")
            return source

        return await chunker.chunk_source(source, config)

    async def chunk_sources(
        self,
        sources: list[SourceFull],
        config: ChunkerConfig | None = None,
    ) -> list[SourceFull]:
        chunker_type = config.preferred_chunker_type or self.default_chunker
        chunker = self.get_chunker(chunker_type)

        if not chunker:
            logger.error(f"Chunker not found: {chunker_type}")
            return sources

        semaphore = asyncio.Semaphore(2)

        try:

            async def chunk_single_source(
                source: SourceFull,
            ) -> SourceFull:
                async with semaphore:
                    try:
                        return await chunker.chunk_source(source, config)
                    except Exception as e:
                        logger.error(f"Failed to chunk source {source.source_id}: {e}")
                        return source

            chunked_sources = await asyncio.gather(
                *[chunk_single_source(s) for s in sources]
            )

            logger.info(f"Chunked {len(chunked_sources)} sources successfully")
            return chunked_sources

        except Exception as e:
            logger.error(f"Chunking multiple sources failed: {e}")
            raise
