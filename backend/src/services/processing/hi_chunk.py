from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from chonkie import (
    LateChunker,
    NeuralChunker,
    SentenceChunker,
    SlumberChunker,
    TokenChunker,
)
from chonkie.refinery import EmbeddingsRefinery
from chonkie.types import Chunk
from loguru import logger
from pydantic import BaseModel

from backend.src.domain.enums import ChunkerType
from backend.src.domain.schemas.config import ChunkerConfig
from backend.src.domain.schemas.doc import _BaseModelFlex


class HierarchicalChunk(_BaseModelFlex):
    """a chunk in the hierarchical tree produced by hichunk."""

    text: str
    token_count: int
    level: int  # 1 = coarsest section, 2 = subsection, …
    start_char: int = 0
    end_char: int = 0
    children: list["HierarchicalChunk"] = field(default_factory=list)
    parent: Optional["HierarchicalChunk"] = None
    embedding: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        snippet = self.text[:60].replace("\n", " ")
        return f"HierarchicalChunk(level={self.level}, tokens={self.token_count}, text='{snippet}…')"


class FlatChunk(_BaseModelFlex):
    """fixed-size leaf chunk used for embedding & retrieval (HC200 step)."""

    text: str
    token_count: int
    start_char: int
    end_char: int
    parent: Optional[HierarchicalChunk] = None
    embedding: Optional[np.ndarray] = None


class HiChunkStructurer:
    """Converts a str document into a hierarchy of HierarchicalChunks"""

    def __init__(self, config: ChunkerConfig):
        #  sentence splitter (step 0 in the paper: S[1:N])
        self.sentence_chunker = SentenceChunker(**config.sentence_config.model_dump())
        self.config = config

        # fixed-size sub-chunker (HC200 step)
        self.token_chunker = TokenChunker(**config.token_config.model_dump())
        self.l1_chunker: SlumberChunker | NeuralChunker | LateChunker

        #  level-1 (coarse) boundary detector
        match config.preferred_strategy:
            case ChunkerType.SLUMBER:
                # llm decides semantic boundaries at paragraph level.
                if SlumberChunker:
                    self.l1_chunker = SlumberChunker(
                        **config.slumber_config.model_dump()
                    )
                    # level-2 uses window to find sub-section breaks
                    self.l2_chunker = SlumberChunker(
                        genie=config.slumber_config.genie,
                        candidate_size=config.slumber_config.candidate_size // 2,
                        min_characters_per_chunk=128,
                    )
                else:
                    raise ImportError("Install chonkie with SlumberChunker support.")

            case ChunkerType.NEURAL:
                if NeuralChunker:
                    # neuralchunker token-classification to predict segment boundaries
                    self.l1_chunker = NeuralChunker(**config.neural_config.model_dump())

                    self.l2_chunker = NeuralChunker(
                        model=config.neural_config.model,
                        min_characters_per_chunk=128,
                    )
                else:
                    raise ImportError("Install chonkie with NeuralChunker support.")

            case _:
                # latechunker: encodes the full document first, then finds splits. preserves global doc context
                from chonkie import SemanticChunker

                self.l1_chunker = LateChunker(**config.late_config.model_dump())

                # l2: smaller semantic chunks inside each l1 section
                self.l2_chunker = SemanticChunker(**config.semantic_config.model_dump())

    # public

    def structure(self, document: str) -> list[HierarchicalChunk]:
        """entry point. returns a list of level-1 HierarchicalChunk roots, each potentially containing level-2 (and deeper) children. iterative inference (algorithm 1) for long documents."""

        # estimate token count via sentence chunker tokenizer
        doc_tokens = self._estimate_tokens(document)

        if doc_tokens <= self.config.max_inference_tokens:
            return self._structure_window(document)
        else:
            return self._iterative_structure(document)

    # helpers

    def _estimate_tokens(self, text: str) -> int:
        # TODO: SWITCH WITH TIKTOKEN
        return len(text) // 4

    def _structure_window(self, text: str) -> list[HierarchicalChunk]:
        """structure a single window that fits in the inference budget."""

        # step 1: get coarse (l1) chunks via the primary chunker
        l1_raw = self.l1_chunker.chunk(text)

        l1_chunks: list[HierarchicalChunk] = []

        if not isinstance(l1_raw, list):
            l1_raw = [l1_raw]

        for raw in l1_raw:
            l1 = HierarchicalChunk(
                text=raw.text,
                token_count=raw.token_count,
                level=1,
                start_char=raw.start_index,
                end_char=raw.end_index,
            )

            # step 2: sub-chunk each l1 into l2 (if max_levels >= 2)
            if self.config.max_levels >= 2 and len(raw.text) > 256:
                l2_raw = self.l2_chunker.chunk(raw.text)
                char_offset = raw.start_index
                for sub in l2_raw:
                    l2 = HierarchicalChunk(
                        text=sub.text,
                        token_count=sub.token_count,
                        level=2,
                        start_char=char_offset + sub.start_index,
                        end_char=char_offset + sub.end_index,
                        parent=l1,
                    )
                    l1.children.append(l2)

                    # step 3: l3 within each l2 (if requested)
                    if self.config.max_levels >= 3 and len(sub.text) > 128:
                        l3_raw = self.token_chunker.chunk(sub.text)
                        sub_offset = char_offset + sub.start_index
                        for seg in l3_raw:
                            l3 = HierarchicalChunk(
                                text=seg.text,
                                token_count=seg.token_count,
                                level=3,
                                start_char=sub_offset + seg.start_index,
                                end_char=sub_offset + seg.end_index,
                                parent=l2,
                            )
                            l2.children.append(l3)

            l1_chunks.append(l1)

        return l1_chunks

    def _iterative_structure(self, document: str) -> list[HierarchicalChunk]:
        """
        algorithm 1 : iterate over the document in windows of max_inference_tokens, merge local chunk points into global ones.
        uses SentenceChunker to split into sentence-bounded windows, then calls _structure_window on each, and merges the resulting trees.
        """

        # split document into sentence-bounded windows
        windows = self.sentence_chunker.chunk(document)

        global_chunks: list[HierarchicalChunk] = []
        residual_text: str = ""

        for win in windows:
            # prepend any residual text from previous window to guide the model
            window_text = (
                (residual_text + " " + win.text).strip() if residual_text else win.text
            )

            local_chunks = self._structure_window(window_text)

            if not local_chunks:
                continue

            if len(local_chunks) >= 2:
                # keep all but the last l1 chunk (which may be incomplete)

                global_chunks.extend(local_chunks[:-1])

                # carry the last chunk's text forward as residual context
                residual_text = local_chunks[-1].text
            else:
                # only one chunk found — entire window is one section
                # set residual to empty
                global_chunks.extend(local_chunks)
                residual_text = ""

        # Flush any remaining residual
        if residual_text.strip():
            flush = self._structure_window(residual_text)
            global_chunks.extend(flush)

        return global_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FLAT CHUNK PRODUCER  (HC200 step)
# ══════════════════════════════════════════════════════════════════════════════


class FlatChunkProducer:
    """
    takes the hierarchical tree and produces fixed-size leaf FlatChunks while preserving the parent pointer for Auto-Merge.
    HC200 step: apply a TokenChunker of size 200 on top of each leaf HierarchicalChunk to normalise semantic granularity.
    """

    def __init__(self, config: ChunkerConfig):
        self.token_chunker = TokenChunker(**config.token_config.model_dump())

    def produce(self, hi_chunks: list[HierarchicalChunk]) -> list[FlatChunk]:
        flat: list[FlatChunk] = []
        for h in hi_chunks:
            flat.extend(self._flatten(h))
        return flat

    def _flatten(self, node: HierarchicalChunk) -> list[FlatChunk]:
        if node.is_leaf:
            raw = self.token_chunker.chunk(node.text)
            result = []
            for r in raw:
                result.append(
                    FlatChunk(
                        text=r.text,
                        token_count=r.token_count,
                        start_char=node.start_char + r.start_index,
                        end_char=node.start_char + r.end_index,
                        parent=node,
                    )
                )
            return result
        else:
            result = []
            for child in node.children:
                result.extend(self._flatten(child))
            return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EMBEDDING ENRICHMENT  (LateChunker / EmbeddingsRefinery)
# ══════════════════════════════════════════════════════════════════════════════
class ChunkEmbedder:
    """adds document-level-aware embeddings to FlatChunks using Chonkie's LateChunker strategy (embed the whole doc first, slice per chunk)."""

    def __init__(
        self,
        embedding_model: str = "nomic-ai/modernbert-embed-base",
    ):
        self.embedding_model = embedding_model
        # We use it for the fallback path.
        self.refinery = EmbeddingsRefinery(
            embedding_model=embedding_model,
        )
        self._late_chunker = None  # lazy-initialise

    def embed(self, document: str, flat_chunks: list[FlatChunk]) -> list[FlatChunk]:
        """Attach embeddings to every FlatChunk in-place and return them."""
        # Build minimal Chonkie Chunk objects so EmbeddingsRefinery can work

        # TODO:  SET TO EMBEDDER SERVICE EMBED
        from chonkie.types import Chunk

        proxy_chunks = [
            Chunk(
                text=fc.text,
                start_index=fc.start_char,
                end_index=fc.end_char,
                token_count=fc.token_count,
            )
            for fc in flat_chunks
        ]

        enriched = self.refinery(proxy_chunks)

        for fc, ec in zip(flat_chunks, enriched):
            if ec.embedding is not None:
                fc.embedding = np.array(ec.embedding)

        return flat_chunks


# 5.  AUTO-MERGE RETRIEVER  (algorithm 2 from the paper)
class AutoMergeRetriever:
    """algorithm 2 (Auto-Merge retrieval) from the HiChunk paper."""

    def __init__(
        self,
        token_budget: int = 4096,
        embedding_model: str = "nomic-ai/modernbert-embed-base",
    ):
        self.token_budget = token_budget
        self._embed_fn = self._build_embed_fn(embedding_model)

    def retrieve(
        self,
        query: str,
        flat_chunks: list[FlatChunk],
    ) -> str:
        """returns the assembled retrieval context as a plain string."""
        if not flat_chunks:
            return ""

        # step 1: embed query and rank chunks

        q_emb = self._embed_fn([query])[0]
        scored = self._rank(q_emb, flat_chunks)

        # step 2: Auto-Merge traversal

        node_ret: list[HierarchicalChunk | FlatChunk] = []
        tk_cur = 0

        for flat_chunk, _score in scored:
            if tk_cur >= self.token_budget:
                break

            node_ret.append(flat_chunk)
            tk_cur = self._context_tokens(node_ret)

            # try merging upward

            parent = flat_chunk.parent
            while parent is not None:
                if not self._cond1(parent, node_ret):
                    break
                if not self._cond2(parent, node_ret, tk_cur):
                    break
                if not self._cond3(parent, tk_cur):
                    break

                # merge: replace covered children with parent

                node_ret = self._merge(node_ret, parent)
                tk_cur = self._context_tokens(node_ret)

                # walk further up the tree

                parent = getattr(parent, "parent", None)

            if tk_cur >= self.token_budget:
                break

        return self._build_context(node_ret)

    #  condition helpers

    def _cond1(
        self,
        parent: HierarchicalChunk,
        node_ret: list,
    ) -> bool:
        """at least 2 of parent's children are already in node_ret."""
        covered = sum(
            1
            for n in node_ret
            if isinstance(n, FlatChunk)
            and n.parent is parent
            or isinstance(n, HierarchicalChunk)
            and n.parent is parent
        )
        return covered >= 2

    def _cond2(
        self,
        parent: HierarchicalChunk,
        node_ret: list,
        tk_cur: int,
    ) -> bool:
        """
        text length of already-recalled children ≥ adaptive threshold θ*.
        θ*(tkcur, p) = len(p)/3 * (1 + tkcur/T)
        grows from len(p)/3 to 2*len(p)/3 as budget fills.
        """

        child_len = sum(
            len(n.text)
            for n in node_ret
            if (isinstance(n, FlatChunk) and n.parent is parent)
            or (isinstance(n, HierarchicalChunk) and n.parent is parent)
        )

        theta_star = (len(parent.text) / 3) * (1 + tk_cur / self.token_budget)

        return child_len >= theta_star

    def _cond3(self, parent: HierarchicalChunk, tk_cur: int) -> bool:
        """remaining budget can fit the parent's full text."""

        parent_tokens = len(parent.text) // 4  # rough estimate

        return (self.token_budget - tk_cur) >= parent_tokens

    # ── Merge / context helpers ────────────────────────────────────────────────

    def _merge(
        self,
        node_ret: list,
        parent: HierarchicalChunk,
    ) -> list:
        """replace children covered by parent with parent itself."""

        new_ret = [
            n
            for n in node_ret
            if not (
                (isinstance(n, FlatChunk) and n.parent is parent)
                or (isinstance(n, HierarchicalChunk) and n.parent is parent)
            )
        ]

        new_ret.append(parent)

        return new_ret

    def _context_tokens(self, node_ret: list) -> int:
        return sum(len(n.text) // 4 for n in node_ret)

    def _build_context(self, node_ret: list) -> str:
        # deduplicate and sort by original position

        seen_texts = set()

        unique = []

        for n in node_ret:
            if n.text not in seen_texts:
                seen_texts.add(n.text)
                unique.append(n)

        # sort by start position if available

        unique.sort(key=lambda n: getattr(n, "start_char", 0))

        return "\n\n".join(n.text for n in unique)

    def _rank(
        self, q_emb: np.ndarray, flat_chunks: list[FlatChunk]
    ) -> list[tuple[FlatChunk, float]]:
        scored = []

        for fc in flat_chunks:
            if fc.embedding is not None:
                sim = self._cosine(q_emb, fc.embedding)
            else:
                sim = 0.0

            scored.append((fc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    @staticmethod
    def _build_embed_fn(model_name: str):
        """Returns a callable (texts: list[str]) -> list[np.ndarray]."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

            def embed(texts):
                return model.encode(texts, normalize_embeddings=True)

            return embed
        except ImportError:
            # fallback: random embeddings (for testing without sentence-transformers)
            import warnings

            warnings.warn(
                "sentence-transformers not installed. Using random embeddings — "
                "retrieval quality will be random. Install with: pip install sentence-transformers"
            )

            def embed(texts):
                return [np.random.randn(384).astype(np.float32) for _ in texts]

            return embed


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FULL HICHUNK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


class HiChunkPipeline:
    """
    HiChunk + Auto-Merge.

    # index a document (done once per document)
    pipeline.index(document_text)

    # retrieve context for a query
    context = pipeline.retrieve(query, token_budget=4096)
    """

    def __init__(self, config: ChunkerConfig):
        self.structurer = HiChunkStructurer(config=config)

        self.flat_producer = FlatChunkProducer(
            chunk_size=config.fixed_chunk_size,
            tokenizer=config.tokenizer,
        )
        self.embedder = ChunkEmbedder(embedding_model=config.embedding_model)

        self.retriever = AutoMergeRetriever(
            token_budget=config.token_budget,
            embedding_model=config.embedding_model,
        )

        self._flat_chunks: list[FlatChunk] = []

        self._hi_chunks: list[HierarchicalChunk] = []

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index(self, document: str) -> None:
        """build the hierarchical structure and embed all flat chunks."""

        logger.info("[HiChunk] Step 1/3 — Hierarchical structuring …")
        # ── Structure ──────────────────────────────────────────────────────────────
        self._hi_chunks = self.structurer.structure(document)

        logger.info(f"[HiChunk]   → {len(self._hi_chunks)} L1 sections found.")
        total_l2 = sum(len(h.children) for h in self._hi_chunks)
        logger.info(f"[HiChunk]   → {total_l2} L2 sub-sections found.")

        logger.info("[HiChunk] Step 2/3 — Fixed-size sub-chunking (HC200) …")
        # ── Produce ──────────────────────────────────────────────────────────────
        self._flat_chunks = self.flat_producer.produce(self._hi_chunks)

        logger.info(f"[HiChunk]   → {len(self._flat_chunks)} flat chunks produced.")

        logger.info("[HiChunk] Step 3/3 — Embedding flat chunks …")
        # ── Embed ──────────────────────────────────────────────────────────────
        self._flat_chunks = self.embedder.embed(document, self._flat_chunks)

        logger.info("[HiChunk] Indexing complete. ✓")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, token_budget: int | None = None) -> str:
        """retrieve the most relevant context for a query using Auto-Merge."""

        if not self._flat_chunks:
            raise RuntimeError("Call .index(document) before .retrieve().")

        budget = token_budget or self.retriever.token_budget
        old_budget = self.retriever.token_budget
        self.retriever.token_budget = budget

        context = self.retriever.retrieve(query, self._flat_chunks)

        self.retriever.token_budget = old_budget
        return context

    #  inspection helpers

    def print_tree(self, max_nodes: int = 20) -> None:
        """Print a compact view of the hierarchical structure."""
        count = 0
        for h in self._hi_chunks:
            if count >= max_nodes:
                print("  … (truncated)")
                break
            snippet = h.text[:80].replace("\n", " ")
            print(
                f"[L1] {snippet}…  ({h.token_count} tokens, {len(h.children)} children)"
            )
            count += 1
            for c in h.children[:3]:
                if count >= max_nodes:
                    break
                sub = c.text[:60].replace("\n", " ")
                print(
                    f"  [L2] {sub}…  ({c.token_count} tokens, {len(c.children)} sub-chunks)"
                )
                count += 1


# ══════════════════════════════════════════════════════════════════════════════
# 7.  QUICK-START DEMO
# ══════════════════════════════════════════════════════════════════════════════

DEMO_DOCUMENT = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to
learn and improve from experience without being explicitly programmed. It focuses
on developing computer programs that can access data and use it to learn for themselves.

The process begins with observations or data, such as examples, direct experience,
or instruction. Machine learning algorithms use computational methods to learn
information directly from data without relying on a predetermined equation as a model.

Types of Machine Learning

Supervised Learning

In supervised learning, the algorithm is trained on labeled data. The model learns
a mapping from inputs to outputs based on example input-output pairs. Common
algorithms include linear regression, decision trees, support vector machines, and
neural networks. The goal is to infer a function from labeled training data.

Applications of supervised learning include email spam detection, image classification,
speech recognition, and medical diagnosis. The training data consists of examples
where the desired output is already known.

Unsupervised Learning

Unsupervised learning involves training on unlabeled data. The algorithm tries to
learn the underlying structure or distribution in the data. Clustering and dimensionality
reduction are common tasks. K-means clustering, hierarchical clustering, and principal
component analysis (PCA) are widely used.

The key challenge in unsupervised learning is evaluating results without ground truth
labels. Applications include customer segmentation, anomaly detection, and
recommendation systems.

Reinforcement Learning

Reinforcement learning is about training agents to make sequences of decisions.
The agent learns by interacting with an environment, receiving rewards for good
actions and penalties for bad ones. Deep Q-networks (DQN) and policy gradient
methods are popular approaches.

This paradigm is especially powerful for sequential decision-making problems such
as game playing, robotic control, and autonomous driving. AlphaGo and ChatGPT both
leverage reinforcement learning components.

Neural Networks and Deep Learning

Deep learning uses neural networks with many layers (deep neural networks) to learn
representations of data. Convolutional neural networks (CNNs) excel at image tasks,
while recurrent neural networks (RNNs) and transformers are powerful for sequential
data like text and time series.

The transformer architecture, introduced in "Attention is All You Need" (2017), has
revolutionised natural language processing. Models like BERT, GPT, and T5 are all
built on the transformer backbone and have achieved state-of-the-art results across
a wide range of NLP benchmarks.

Training and Evaluation

Model training involves optimising parameters to minimise a loss function, typically
using gradient descent and backpropagation. Regularisation techniques such as dropout,
weight decay, and early stopping help prevent overfitting on training data.

Evaluation metrics depend on the task: accuracy and F1 score for classification,
RMSE and MAE for regression, BLEU and ROUGE for generation tasks. Cross-validation
provides a robust estimate of generalisation performance.
"""


def demo():
    print("=" * 60)
    print("HiChunk + Chonkie — Quick Demo")
    print("=" * 60)

    # Using "semantic" strategy: no API key or GPU required
    pipeline = HiChunkPipeline(config=ChunkerConfig())

    pipeline.index(DEMO_DOCUMENT)

    print()
    pipeline.print_tree()

    queries = [
        "What are the main types of machine learning?",
        "How does reinforcement learning work and what are its applications?",
        "What is the transformer architecture and why is it important?",
    ]

    print()
    print("─" * 60)
    for query in queries:
        print(f"\nQuery: {query}")
        context = pipeline.retrieve(query, token_budget=512)
        print(f"Retrieved context ({len(context)} chars):")
        print(context[:400] + ("…" if len(context) > 400 else ""))
        print("─" * 60)


if __name__ == "__main__":
    demo()
