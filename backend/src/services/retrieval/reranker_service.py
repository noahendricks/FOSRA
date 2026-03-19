from __future__ import annotations

from typing import TYPE_CHECKING

from flashrank import Ranker, RerankRequest
from loguru import logger

if TYPE_CHECKING:
    from backend.src.domain.schemas.config import RerankerConfig
    from backend.src.services.retrieval.vector_service import RetrievedChunk


# Module-level singleton — loaded once per process.
_ranker_cache: dict[str, Ranker] = {}


def _get_ranker(model_name: str, cache_dir: str | None = None) -> Ranker:
    """Return a cached FlashRank ``Ranker``, creating one on first call."""
    if model_name not in _ranker_cache:
        kwargs: dict = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        logger.info("Loading FlashRank model: {}", model_name)
        _ranker_cache[model_name] = Ranker(**kwargs)
    return _ranker_cache[model_name]


class RerankerService:
    """Cross-encoder reranking via FlashRank.

    Default model: ``ms-marco-MiniLM-L-12-v2`` (~33 MB, CPU-only, <50 ms
    for typical batches).
    """

    DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"

    def __init__(self, config: RerankerConfig | None = None):
        from backend.src.domain.schemas.config import RerankerConfig as _RC

        self._config = config or _RC()
        self._model_name: str = self._config.model or self.DEFAULT_MODEL
        self._ranker: Ranker = _get_ranker(self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank *chunks* against *query* using FlashRank.

        Parameters
        ----------
        query:
            The user query (or reformulated query).
        chunks:
            Retrieved chunks from vector search.
        top_k:
            Maximum number of results to return.  Falls back to
            ``RerankerConfig.top_k`` (default 10).
        score_threshold:
            Minimum reranker score to keep.  Falls back to
            ``RerankerConfig.score_threshold`` (default ``None`` = keep all).

        Returns
        -------
        list[RetrievedChunk]
            Reranked (and optionally filtered) chunks, sorted by descending
            reranker score.  Each chunk's ``.score`` is **replaced** with the
            cross-encoder score so downstream consumers see the reranked
            ordering.
        """
        if not chunks:
            return []

        top_k = top_k or self._config.top_k or 10
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self._config.score_threshold
        )

        # Build FlashRank passages — id is the list index so we can map back.
        passages = [
            {
                "id": idx,
                "text": chunk.text,
                "meta": {"original_score": chunk.score},
            }
            for idx, chunk in enumerate(chunks)
        ]

        request = RerankRequest(query=query, passages=passages)
        ranked: list[dict] = self._ranker.rerank(request)

        # Map results back to RetrievedChunk objects.
        reranked: list[RetrievedChunk] = []
        for item in ranked:
            idx = item["id"]
            rerank_score: float = item["score"]

            if score_threshold is not None and rerank_score < score_threshold:
                continue

            chunk = chunks[idx].model_copy()
            chunk.score = rerank_score
            reranked.append(chunk)

        # FlashRank returns results sorted desc, but enforce it.
        reranked.sort(key=lambda c: c.score, reverse=True)

        return reranked[:top_k]
