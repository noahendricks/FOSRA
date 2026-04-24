from __future__ import annotations

from chonkie import (
    CodeChunker,
    LateChunker,
    NeuralChunker,
    RecursiveChunker,
    SentenceChunker,
    SentenceTransformerEmbeddings,
    TokenChunker,
)
from loguru import logger

from backend.src.domain.enums import ChunkerType
from backend.src.domain.schemas.doc import (
    Doc,
    HierarchicalChunk,
    SectionMetadata,
    Subsection,
)
from backend.src.settings import ChunkerConfig


class FlatChunkProducer:
    def __init__(self, config: ChunkerConfig):
        self.token_chunker = TokenChunker(**config.token_config.model_dump())

    def produce(self, hi_chunks: list[HierarchicalChunk]) -> list[Subsection]:
        flat: list[Subsection] = []
        for h in hi_chunks:
            flat.extend(self._flatten(h))
        return flat

    def _flatten(self, node: HierarchicalChunk) -> list[Subsection]:
        result: list[Subsection] = []

        if not node.is_leaf:
            for child in node.children:
                result.extend(self._flatten(child))
            return result

        if node.token_count >= 5 and node.text.strip() != "":
            meta = SectionMetadata(
                token_count=node.token_count,
                start_char=node.start_char,
                end_char=node.end_char,
            )
            result.append(Subsection(text=node.text, metadata=meta))

        return result


class HiChunkStructurer:
    """Converts a str document into a hierarchy of HierarchicalChunks"""

    def __init__(self, config: ChunkerConfig):
        #  sentence splitter (step 0 in the paper: S[1:N])
        self.config = config
        self.sentence_chunker = SentenceChunker(**config.sentence_config.model_dump())
        self.token_chunker = TokenChunker(**config.token_config.model_dump())
        self.l1_chunker: CodeChunker | NeuralChunker | LateChunker
        self.flat_producer = FlatChunkProducer(config=config)

        #  level-1 (coarse) boundary detector
        match config.preferred_strategy:
            case ChunkerType.CODE:
                # llm decides semantic boundaries at paragraph level.

                self.l2_chunker = CodeChunker(chunk_size=1200)
                # level-2 uses window to find sub-section breaks

                self.l1_chunker = NeuralChunker(
                    model=config.neural_config.model,
                    min_characters_per_chunk=128,
                )

            case ChunkerType.NEURAL:
                if NeuralChunker:
                    # neuralchunker token-classification to predict segment boundaries
                    self.l1_chunker = NeuralChunker(**config.neural_config.model_dump())

                    # L2: RecursiveChunker with ~512 token target, from config
                    from chonkie import RecursiveRules

                    rec_dump = config.recursive_config.model_dump(
                        exclude={"chunk_overlap"}
                    )
                    if isinstance(rec_dump.get("rules"), dict):
                        rec_dump["rules"] = RecursiveRules.from_dict(rec_dump["rules"])
                    self.l2_chunker = RecursiveChunker(**rec_dump)
                else:
                    raise ImportError("Install chonkie with NeuralChunker support.")

            case _:
                from chonkie import SemanticChunker

                late_embedding_model = SentenceTransformerEmbeddings(
                    model=config.late_config.embedding_model, trust_remote_code=True
                )
                self.l1_chunker = LateChunker(
                    embedding_model=late_embedding_model,
                    **config.late_config.model_dump(exclude={"embedding_model"}),
                )

                # l2: smaller semantic chunks inside each l1 section
                semantic_embedding_model = SentenceTransformerEmbeddings(
                    config.semantic_config.embedding_model, trust_remote_code=True
                )

                self.l2_chunker = SemanticChunker(
                    embedding_model=semantic_embedding_model,
                    **config.semantic_config.model_dump(exclude={"embedding_model"}),
                )

    # public
    def structure(self, document: Doc, /) -> list[HierarchicalChunk]:
        """entry point."""

        # estimate token count via sentence chunker tokenizer
        return self._structure_window(document)

    def _structure_window(self, doc: Doc, /) -> list[HierarchicalChunk]:
        """structure a single window that fits in the inference budget."""

        # step 1: get coarse (l1) chunks via the primary chunker
        l1_raw = self.l1_chunker.chunk(doc.page_content)

        l1_chunks: list[HierarchicalChunk] = []

        # WARN: CHECK IF METADATA [ESPECIALLY CODE] MAKES SENSE HIERARCHICALLY
        if not isinstance(l1_raw, list):
            l1_raw = [l1_raw]

        for raw in l1_raw:
            l1 = HierarchicalChunk(
                text=raw.text,
                token_count=raw.token_count,
                level=1,
                start_char=raw.start_index,
                end_char=raw.end_index,
                metadata=doc.metadata,
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
                        metadata=doc.metadata,
                    )
                    l1.children.append(l2)

                    l2.parent = l1

                    # step 3: l3 within each l2 (if requested)
                    if self.config.max_levels >= 3 and len(sub.text) > 128:
                        l3_raw = self.token_chunker.chunk(sub.text)

                        if len(l3_raw) > 1 or (
                            len(l3_raw) == 1 and l3_raw[0].text != sub.text
                        ):
                            sub_offset = char_offset + sub.start_index
                            for seg in l3_raw:
                                l3 = HierarchicalChunk(
                                    text=seg.text,
                                    token_count=seg.token_count,
                                    level=3,
                                    start_char=sub_offset + seg.start_index,
                                    end_char=sub_offset + seg.end_index,
                                    metadata=doc.metadata,
                                )
                                l2.children.append(l3)
                                l3.parent = l2

            l1_chunks.append(l1)

        return l1_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FULL HICHUNK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


class HiChunk:
    @staticmethod
    def index(
        document: Doc,
        structurer: HiChunkStructurer,
    ) -> list[Subsection]:
        """build the hierarchical structure and embed all flat chunks."""

        logger.info("[HiChunk] Step 1/3 — Hierarchical structuring …")

        # HI Chunks
        _hi_chunks = structurer.structure(document)

        logger.info("[HiChunk]   → {} L1 sections found.", len(_hi_chunks))

        # L2 Count
        total_l2 = sum(len(h.children) for h in _hi_chunks)

        logger.info("[HiChunk]   → {} L2 sub-sections found.", total_l2)

        logger.info("[HiChunk] Step 2/3 — Fixed-size sub-chunking (HC200) …")

        # Flat Chunks
        _flat_chunks = structurer.flat_producer.produce(_hi_chunks)

        logger.info("[HiChunk]   → {} flat chunks produced.", len(_flat_chunks))

        return _flat_chunks

    @staticmethod
    def _print_tree(hi_chunks: list[HierarchicalChunk], max_nodes: int = 20) -> None:
        """Print a compact view of the hierarchical structure."""
        count = 0
        for h in hi_chunks:
            if count >= max_nodes:
                logger.debug("  … (truncated)")
                break
            snippet = h.text[:80].replace("\n", " ")
            logger.debug(
                "[L1] {}…  ({} tokens, {} children)",
                snippet,
                h.token_count,
                len(h.children),
            )
            count += 1
            for c in h.children[:3]:
                if count >= max_nodes:
                    break
                sub = c.text[:60].replace("\n", " ")
                logger.debug(
                    "  [L2] {}…  ({} tokens, {} sub-chunks)",
                    sub,
                    c.token_count,
                    len(c.children),
                )
                count += 1
