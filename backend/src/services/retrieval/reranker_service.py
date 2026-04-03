from __future__ import annotations

from typing import TYPE_CHECKING

from flashrank import Ranker, RerankRequest
from loguru import logger

if TYPE_CHECKING:
    from backend.src.settings import RerankerConfig
    from backend.src.services.retrieval.vector_service import RetrievedChunk

from backend.src.domain.enums import RerankerType
from backend.src.domain.schemas.graph import CodeNode


# =============================================================================
# FlashRank reranker
# =============================================================================
_flashrank_cache: dict[str, Ranker] = {}


def _get_flashrank_ranker(model_name: str, cache_dir: str | None = None) -> Ranker:
    if model_name not in _flashrank_cache:
        kwargs: dict = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        logger.info("Loading FlashRank model: {}", model_name)
        _flashrank_cache[model_name] = Ranker(**kwargs)
    return _flashrank_cache[model_name]


# =============================================================================
# FlagEmbedding BGE reranker
# =============================================================================
_bgereranker_cache: dict[str, "FlagReranker"] = {}  # type: ignore[name-defined]


def _get_bge_reranker(model_name: str) -> "FlagReranker":  # type: ignore[name-defined]
    from FlagEmbedding import FlagReranker

    if model_name not in _bgereranker_cache:
        logger.info("Loading BGE reranker model: {}", model_name)
        _bgereranker_cache[model_name] = FlagReranker(model_name, use_fp16=False)
    return _bgereranker_cache[model_name]


# =============================================================================
# RerankerService — dispatches to Flag or FlashRank based on config
# =============================================================================
class RerankerService:
    """Cross-encoder reranking with pluggable backend.

    Default backend: BGE reranker (``BAAI/bge-reranker-v2-m3``).
    Fallback: FlashRank (``ms-marco-MiniLM-L-12-v2``).
    """

    def __init__(self, config: RerankerConfig | None = None):
        from backend.src.settings import RerankerConfig as _RC

        self._config = config or _RC()
        self._provider = self._config.rerank_provider
        self._flashrank_model: str = self._config.model or "ms-marco-MiniLM-L-12-v2"
        self._bge_model: str = self._config.bge_model or "BAAI/bge-reranker-v2-m3"

    def _rank_bge(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        score_threshold: float | None,
    ) -> list[RetrievedChunk]:
        reranker = _get_bge_reranker(self._bge_model)

        pairs = [[query, chunk.text] for chunk in chunks]
        raw_scores = reranker.compute_score(pairs, normalize=True)

        scored: list[tuple[int, float]] = [
            (idx, score) for idx, score in enumerate(raw_scores)
        ]
        if score_threshold is not None:
            scored = [(idx, s) for idx, s in scored if s >= score_threshold]

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        reranked: list[RetrievedChunk] = []
        for idx, score in scored:
            chunk = chunks[idx].model_copy()
            chunk.score = score
            reranked.append(chunk)

        return reranked

    def _rank_flashrank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        score_threshold: float | None,
    ) -> list[RetrievedChunk]:
        reranker = _get_flashrank_ranker(self._flashrank_model)

        passages = [
            {"id": idx, "text": chunk.text, "meta": {"original_score": chunk.score}}
            for idx, chunk in enumerate(chunks)
        ]

        request = RerankRequest(query=query, passages=passages)
        ranked: list[dict] = reranker.rerank(request)

        reranked: list[RetrievedChunk] = []
        for item in ranked:
            idx = item["id"]
            rerank_score: float = item["score"]

            if score_threshold is not None and rerank_score < score_threshold:
                continue

            chunk = chunks[idx].model_copy()
            chunk.score = rerank_score
            reranked.append(chunk)

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_k]

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank *chunks* against *query* using the configured backend.

        Parameters
        ----------
        query:
            The user query (or reformulated query).
        chunks:
            Retrieved chunks from vector search.
        top_k:
            Maximum number of results to return.
        score_threshold:
            Minimum reranker score to keep.

        Returns
        -------
        list[RetrievedChunk]
            Reranked chunks sorted by descending reranker score.
        """
        if not chunks:
            return []

        top_k = top_k or self._config.top_k or 10
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self._config.score_threshold
        )

        try:
            match self._provider:
                case RerankerType.BGE:
                    return self._rank_bge(query, chunks, top_k, score_threshold)
                case RerankerType.FLASHRANK:
                    return self._rank_flashrank(query, chunks, top_k, score_threshold)
                case _:
                    logger.warning(
                        "Unknown rerank provider '{}', falling back to FlashRank",
                        self._provider,
                    )
                    return self._rank_flashrank(query, chunks, top_k, score_threshold)
        except Exception as e:
            logger.warning(
                "Reranking failed with {} backend: {}. Falling back to FlashRank.",
                self._provider,
                e,
            )
            return self._rank_flashrank(query, chunks, top_k, score_threshold)

    def rerank_code_nodes(
        self,
        query: str,
        nodes: list[CodeNode],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[CodeNode]:
        """Rerank code graph nodes against a query.

        Adapts CodeNode → RetrievedChunk, reranks, returns reranked CodeNodes.
        """
        from backend.src.services.retrieval.vector_service import RetrievedChunk

        if not nodes:
            return []

        top_k = top_k or self._config.top_k or 10
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self._config.score_threshold
        )

        def to_retrieved(node: CodeNode) -> RetrievedChunk:
            content_parts = []
            if node.signature:
                content_parts.append(node._signature_to_string())
            if node.docstring:
                content_parts.append(node.docstring)
            if node.source_code:
                content_parts.append(node.source_code)
            text = "\n\n".join(content_parts) if content_parts else node.name
            return RetrievedChunk(
                text=text,
                token_count=len(text) // 4,
                start_char=0,
                score=0.0,
                payload={
                    "qualified_name": node.qualified_name,
                    "chunk_id": node.qualified_name,
                },
            )

        chunks = [to_retrieved(n) for n in nodes]
        try:
            reranked_chunks = self.rerank(
                query=query,
                chunks=chunks,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.opt(exception=True).warning("Code node reranking failed")
            return nodes[:top_k]

        reranked_qns = {c.payload["qualified_name"] for c in reranked_chunks}
        result: list[CodeNode] = []
        qn_map = {n.qualified_name: n for n in nodes}
        for qn in reranked_qns:
            if qn in qn_map:
                result.append(qn_map[qn])
        return result
