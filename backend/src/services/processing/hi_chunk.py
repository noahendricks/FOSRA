from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from chonkie import (
    CodeChunker,
    LateChunker,
    NeuralChunker,
    SentenceChunker,
    SentenceTransformerEmbeddings,
    TokenChunker,
)
from chonkie.refinery import EmbeddingsRefinery
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic.v1.utils import to_camel
from qdrant_client.models import PointStruct

from backend.src.domain.enums import ChunkerType
from backend.src.domain.schemas.config import ChunkerConfig, EmbedderConfig
from backend.src.domain.schemas.doc import Chunk, ChunkMetadata


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


class RetrievedChunk(_BaseModelFlex):
    text: str
    token_count: int
    start_char: int
    score: float
    payload: dict[str, Any]


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
        self.config = config
        self.sentence_chunker = SentenceChunker(**config.sentence_config.model_dump())
        self.token_chunker = TokenChunker(**config.token_config.model_dump())
        self.l1_chunker: CodeChunker | NeuralChunker | LateChunker

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

                    self.l2_chunker = NeuralChunker(
                        model=config.neural_config.model,
                        min_characters_per_chunk=128,
                    )
                else:
                    raise ImportError("Install chonkie with NeuralChunker support.")

            case _:
                from chonkie import SemanticChunker

                late_embedding_model = SentenceTransformerEmbeddings(
                    model=config.late_config.embedding_model, trust_remote_code=True
                )
                self.l1_chunker = LateChunker(
                    embedding_model=late_embedding_model,
                    **config.late_config.model_dump(exclude=("embedding_model")),
                )

                # l2: smaller semantic chunks inside each l1 section
                semantic_embedding_model = SentenceTransformerEmbeddings(
                    config.semantic_config.embedding_model, trust_remote_code=True
                )

                self.l2_chunker = SemanticChunker(
                    embedding_model=semantic_embedding_model,
                    **config.semantic_config.model_dump(exclude=("embedding_model")),
                )

        print(self.l1_chunker, "l1_chunker")
        print(self.l2_chunker, "l2_chunker")

    # public

    def structure(self, document: str) -> list[HierarchicalChunk]:
        """entry point."""

        # estimate token count via sentence chunker tokenizer
        doc_tokens = self._estimate_tokens(document)

        if doc_tokens <= self.config.max_inference_tokens:
            return self._structure_window(document)
        else:
            return self._iterative_structure(document)

    # helpers

    def _estimate_tokens(self, text: str) -> int:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

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

    def to_domain_chunk(flat_chunks: list[FlatChunk]):
        domain_chunks = []
        for c in flat_chunks:

            meta = ChunkMetadata()

            d_chunk = Chunk()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FULL HICHUNK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


class HiChunk:

    @staticmethod
    def index(
        document: str,
        structurer: HiChunkStructurer,
        flat_producer: FlatChunkProducer,
    ) -> None:
        """build the hierarchical structure and embed all flat chunks."""

        logger.info("[HiChunk] Step 1/3 — Hierarchical structuring …")

        # structure
        _hi_chunks = structurer.structure(document)

        logger.info(f"[HiChunk]   → {len(_hi_chunks)} L1 sections found.")
        total_l2 = sum(len(h.children) for h in _hi_chunks)
        logger.info(f"[HiChunk]   → {total_l2} L2 sub-sections found.")

        logger.info("[HiChunk] Step 2/3 — Fixed-size sub-chunking (HC200) …")

        # produce
        _flat_chunks = flat_producer.produce(_hi_chunks)

        logger.info(f"[HiChunk]   → {len(_flat_chunks)} flat chunks produced.")

        logger.info("[HiChunk] Step 3/3 — Embedding flat chunks …")

        #  embed
        # self._flat_chunks = self.embedder.embed(document, self._flat_chunks)

        print(_hi_chunks)

        print("\n\n")
        for i in _flat_chunks:
            print("-----------------beginning of chunk-----------")
            print("\n")
            print(i)
            print("\n")
            print("-----------------end of chunk-----------")
        logger.info("[HiChunk] Indexing complete. ✓")

    #  inspection helpers

    @staticmethod
    def print_tree(hi_chunks: list[HierarchicalChunk], max_nodes: int = 20) -> None:
        """Print a compact view of the hierarchical structure."""
        count = 0
        for h in hi_chunks:
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

    @staticmethod
    def build_points(
        chunks: list[Chunk],
        source_id: str,
    ) -> list[PointStruct]:
        points = []
        for i, chunk in enumerate(chunks):
            parent = chunk.metadata.parent  # HierarchicalChunk (L2 or L1 leaf)

            # Walk up to get all ancestor text for merge candidates
            grandparent = getattr(parent, "parent", None)  # L1 if parent is L2

            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        "dense": chunk.metadata.dense_embedding,
                        "sparse": chunk.metadata.sparse_embedding,
                        "late-interaction": chunk.metadata.late_embedding,
                    },
                    payload={
                        "text": chunk.text,
                        "source_id": source_id,
                        "token_count": chunk.metadata.token_count,
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        # Hierarchy for Auto-Merge
                        "parent_text": parent.text if parent else None,
                        "parent_token_count": parent.token_count if parent else 0,
                        "parent_start_char": parent.start_char if parent else None,
                        "parent_end_char": parent.end_char if parent else None,
                        "parent_level": parent.level if parent else None,
                        "grandparent_text": grandparent.text if grandparent else None,
                        "grandparent_token_count": grandparent.token_count
                        if grandparent
                        else 0,
                        "grandparent_start_char": grandparent.start_char
                        if grandparent
                        else None,
                        "grandparent_level": grandparent.level if grandparent else None,
                        # For grouping siblings during merge check
                        "parent_id": f"{source_id}:{parent.start_char}:{parent.end_char}"
                        if parent
                        else None,
                        "grandparent_id": f"{source_id}:{grandparent.start_char}:{grandparent.end_char}"
                        if grandparent
                        else None,
                    },
                )
            )

        return points

    @staticmethod
    def auto_merge(
        results: list[RetrievedChunk],
        token_budget: int,
        merge_threshold: float = 0.5,  # fraction of parent's children needed
    ) -> str:
        """
        Auto-Merge retrieval post-processor.
        Takes RRF-ranked Qdrant results, merges up to parent/grandparent
        where coverage conditions are met, returns assembled context string.
        """

        if not results:
            return ""

        # Group retrieved chunks by parent_id
        parent_groups: dict[str, list[RetrievedChunk]] = {}
        no_parent: list[RetrievedChunk] = []

        for chunk in results:
            pid = chunk.payload.get("parent_id")
            if pid:
                parent_groups.setdefault(pid, []).append(chunk)
            else:
                no_parent.append(chunk)

        # Decide what to keep: parent text or individual chunks
        node_ret: list[tuple[str, int, int]] = []  # (text, token_count, start_char)
        tokens_used = 0

        for pid, siblings in parent_groups.items():
            if tokens_used >= token_budget:
                break

            parent_text = siblings[0].payload.get("parent_text")
            parent_tokens = siblings[0].payload.get("parent_token_count", 0)
            parent_start = siblings[0].payload.get("parent_start_char", 0)
            grandparent_id = siblings[0].payload.get("grandparent_id")

            # Cond1: at least 2 siblings retrieved
            cond1 = len(siblings) >= 2

            # Cond2: covered text >= adaptive threshold
            covered_chars = sum(
                (c.payload["end_char"] - c.payload["start_char"]) for c in siblings
            )
            parent_chars = len(parent_text) if parent_text else 1
            theta_star = (parent_chars / 3) * (1 + tokens_used / token_budget)
            cond2 = covered_chars >= theta_star

            # Cond3: parent fits in remaining budget
            cond3 = (token_budget - tokens_used) >= parent_tokens

            if cond1 and cond2 and cond3 and parent_text:
                # Check if we can merge further up to grandparent
                # (only attempt if grandparent exists and budget allows)
                grandparent_text = siblings[0].payload.get("grandparent_text")
                grandparent_tokens = siblings[0].payload.get(
                    "grandparent_token_count", 0
                )

                if (
                    grandparent_text
                    and grandparent_tokens <= (token_budget - tokens_used)
                    and len(siblings) >= 3
                ):  # stricter threshold for grandparent
                    node_ret.append(
                        (
                            grandparent_text,
                            grandparent_tokens,
                            siblings[0].payload.get("grandparent_start_char", 0),
                        )
                    )
                    tokens_used += grandparent_tokens
                else:
                    node_ret.append((parent_text, parent_tokens, parent_start))
                    tokens_used += parent_tokens
            else:
                # Keep individual chunks
                for chunk in siblings:
                    if tokens_used >= token_budget:
                        break
                    node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
                    tokens_used += chunk.token_count

        # Add orphan chunks (no parent)
        for chunk in no_parent:
            if tokens_used >= token_budget:
                break
            node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
            tokens_used += chunk.token_count

        # Deduplicate and sort by position
        seen = set()
        unique = []
        for text, tokens, start in node_ret:
            if text not in seen:
                seen.add(text)
                unique.append((text, tokens, start))

        unique.sort(key=lambda x: x[2])  # sort by start_char = document order

        return "\n\n".join(text for text, _, _ in unique)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  QUICK-START DEMO
# ══════════════════════════════════════════════════════════════════════════════


DEMO_DOCUMENT = """
An Occurrence at Owl Creek Bridge

by Ambrose Bierce

THE MILLENNIUM FULCRUM EDITION, 1988




I


A man stood upon a railroad bridge in northern Alabama, looking down
into the swift water twenty feet below. The man’s hands were behind his
back, the wrists bound with a cord. A rope closely encircled his neck.
It was attached to a stout cross-timber above his head and the slack
fell to the level of his knees. Some loose boards laid upon the ties
supporting the rails of the railway supplied a footing for him and his
executioners—two private soldiers of the Federal army, directed by a
sergeant who in civil life may have been a deputy sheriff. At a short
remove upon the same temporary platform was an officer in the uniform
of his rank, armed. He was a captain. A sentinel at each end of the
bridge stood with his rifle in the position known as “support,” that is
to say, vertical in front of the left shoulder, the hammer resting on
the forearm thrown straight across the chest—a formal and unnatural
position, enforcing an erect carriage of the body. It did not appear to
be the duty of these two men to know what was occurring at the center
of the bridge; they merely blockaded the two ends of the foot planking
that traversed it.

Beyond one of the sentinels nobody was in sight; the railroad ran
straight away into a forest for a hundred yards, then, curving, was
lost to view. Doubtless there was an outpost farther along. The other
bank of the stream was open ground—a gentle slope topped with a
stockade of vertical tree trunks, loopholed for rifles, with a single
embrasure through which protruded the muzzle of a brass cannon
commanding the bridge. Midway up the slope between the bridge and fort
were the spectators—a single company of infantry in line, at “parade
rest,” the butts of their rifles on the ground, the barrels inclining
slightly backward against the right shoulder, the hands crossed upon
the stock. A lieutenant stood at the right of the line, the point of
his sword upon the ground, his left hand resting upon his right.
Excepting the group of four at the center of the bridge, not a man
moved. The company faced the bridge, staring stonily, motionless. The
sentinels, facing the banks of the stream, might have been statues to
adorn the bridge. The captain stood with folded arms, silent, observing
the work of his subordinates, but making no sign. Death is a dignitary
who when he comes announced is to be received with formal
manifestations of respect, even by those most familiar with him. In the
code of military etiquette silence and fixity are forms of deference.

The man who was engaged in being hanged was apparently about
thirty-five years of age. He was a civilian, if one might judge from
his habit, which was that of a planter. His features were good—a
straight nose, firm mouth, broad forehead, from which his long, dark
hair was combed straight back, falling behind his ears to the collar of
his well fitting frock coat. He wore a moustache and pointed beard, but
no whiskers; his eyes were large and dark gray, and had a kindly
expression which one would hardly have expected in one whose neck was
in the hemp. Evidently this was no vulgar assassin. The liberal
military code makes provision for hanging many kinds of persons, and
gentlemen are not excluded.

The preparations being complete, the two private soldiers stepped aside
and each drew away the plank upon which he had been standing. The
sergeant turned to the captain, saluted and placed himself immediately
behind that officer, who in turn moved apart one pace. These movements
left the condemned man and the sergeant standing on the two ends of the
same plank, which spanned three of the cross-ties of the bridge. The
end upon which the civilian stood almost, but not quite, reached a
fourth. This plank had been held in place by the weight of the captain;
it was now held by that of the sergeant. At a signal from the former
the latter would step aside, the plank would tilt and the condemned man
go down between two ties. The arrangement commended itself to his
judgement as simple and effective. His face had not been covered nor
his eyes bandaged. He looked a moment at his “unsteadfast footing,”
then let his gaze wander to the swirling water of the stream racing
madly beneath his feet. A piece of dancing driftwood caught his
attention and his eyes followed it down the current. How slowly it
appeared to move! What a sluggish stream!

He closed his eyes in order to fix his last thoughts upon his wife and
children. The water, touched to gold by the early sun, the brooding
mists under the banks at some distance down the stream, the fort, the
soldiers, the piece of drift—all had distracted him. And now he became
conscious of a new disturbance. Striking through the thought of his
dear ones was sound which he could neither ignore nor understand, a
sharp, distinct, metallic percussion like the stroke of a blacksmith’s
hammer upon the anvil; it had the same ringing quality. He wondered
what it was, and whether immeasurably distant or near by— it seemed
both. Its recurrence was regular, but as slow as the tolling of a death
knell. He awaited each new stroke with impatience and—he knew not
why—apprehension. The intervals of silence grew progressively longer;
the delays became maddening. With their greater infrequency the sounds
increased in strength and sharpness. They hurt his ear like the thrust
of a knife; he feared he would shriek. What he heard was the ticking of
his watch.

He unclosed his eyes and saw again the water below him. “If I could
free my hands,” he thought, “I might throw off the noose and spring
into the stream. By diving I could evade the bullets and, swimming
vigorously, reach the bank, take to the woods and get away home. My
home, thank God, is as yet outside their lines; my wife and little ones
are still beyond the invader’s farthest advance.”

As these thoughts, which have here to be set down in words, were
flashed into the doomed man’s brain rather than evolved from it the
captain nodded to the sergeant. The sergeant stepped aside.




II


Peyton Farquhar was a well to do planter, of an old and highly
respected Alabama family. Being a slave owner and like other slave
owners a politician, he was naturally an original secessionist and
ardently devoted to the Southern cause. Circumstances of an imperious
nature, which it is unnecessary to relate here, had prevented him from
taking service with that gallant army which had fought the disastrous
campaigns ending with the fall of Corinth, and he chafed under the
inglorious restraint, longing for the release of his energies, the
larger life of the soldier, the opportunity for distinction. That
opportunity, he felt, would come, as it comes to all in wartime.
Meanwhile he did what he could. No service was too humble for him to
perform in the aid of the South, no adventure too perilous for him to
undertake if consistent with the character of a civilian who was at
heart a soldier, and who in good faith and without too much
qualification assented to at least a part of the frankly villainous
dictum that all is fair in love and war.

One evening while Farquhar and his wife were sitting on a rustic bench
near the entrance to his grounds, a gray-clad soldier rode up to the
gate and asked for a drink of water. Mrs. Farquhar was only too happy
to serve him with her own white hands. While she was fetching the water
her husband approached the dusty horseman and inquired eagerly for news
from the front.

“The Yanks are repairing the railroads,” said the man, “and are getting
ready for another advance. They have reached the Owl Creek bridge, put
it in order and built a stockade on the north bank. The commandant has
issued an order, which is posted everywhere, declaring that any
civilian caught interfering with the railroad, its bridges, tunnels, or
trains will be summarily hanged. I saw the order.”

“How far is it to the Owl Creek bridge?” Farquhar asked.

“About thirty miles.”

“Is there no force on this side of the creek?”

“Only a picket post half a mile out, on the railroad, and a single
sentinel at this end of the bridge.”

“Suppose a man—a civilian and student of hanging—should elude the
picket post and perhaps get the better of the sentinel,” said Farquhar,
smiling, “what could he accomplish?”

The soldier reflected. “I was there a month ago,” he replied. “I
observed that the flood of last winter had lodged a great quantity of
driftwood against the wooden pier at this end of the bridge. It is now
dry and would burn like tinder.”

The lady had now brought the water, which the soldier drank. He thanked
her ceremoniously, bowed to her husband and rode away. An hour later,
after nightfall, he repassed the plantation, going northward in the
direction from which he had come. He was a Federal scout.




III


As Peyton Farquhar fell straight downward through the bridge he lost
consciousness and was as one already dead. From this state he was
awakened—ages later, it seemed to him—by the pain of a sharp pressure
upon his throat, followed by a sense of suffocation. Keen, poignant
agonies seemed to shoot from his neck downward through every fiber of
his body and limbs. These pains appeared to flash along well defined
lines of ramification and to beat with an inconceivably rapid
periodicity. They seemed like streams of pulsating fire heating him to
an intolerable temperature. As to his head, he was conscious of nothing
but a feeling of fullness—of congestion. These sensations were
unaccompanied by thought. The intellectual part of his nature was
already effaced; he had power only to feel, and feeling was torment. He
was conscious of motion. Encompassed in a luminous cloud, of which he
was now merely the fiery heart, without material substance, he swung
through unthinkable arcs of oscillation, like a vast pendulum. Then all
at once, with terrible suddenness, the light about him shot upward with
the noise of a loud splash; a frightful roaring was in his ears, and
all was cold and dark. The power of thought was restored; he knew that
the rope had broken and he had fallen into the stream. There was no
additional strangulation; the noose about his neck was already
suffocating him and kept the water from his lungs. To die of hanging at
the bottom of a river!—the idea seemed to him ludicrous. He opened his
eyes in the darkness and saw above him a gleam of light, but how
distant, how inaccessible! He was still sinking, for the light became
fainter and fainter until it was a mere glimmer. Then it began to grow
and brighten, and he knew that he was rising toward the surface—knew it
with reluctance, for he was now very comfortable. “To be hanged and
drowned,” he thought, “that is not so bad; but I do not wish to be
shot. No; I will not be shot; that is not fair.”

He was not conscious of an effort, but a sharp pain in his wrist
apprised him that he was trying to free his hands. He gave the struggle
his attention, as an idler might observe the feat of a juggler, without
interest in the outcome. What splendid effort!—what magnificent, what
superhuman strength! Ah, that was a fine endeavor! Bravo! The cord fell
away; his arms parted and floated upward, the hands dimly seen on each
side in the growing light. He watched them with a new interest as first
one and then the other pounced upon the noose at his neck. They tore it
away and thrust it fiercely aside, its undulations resembling those of
a water snake. “Put it back, put it back!” He thought he shouted these
words to his hands, for the undoing of the noose had been succeeded by
the direst pang that he had yet experienced. His neck ached horribly;
his brain was on fire, his heart, which had been fluttering faintly,
gave a great leap, trying to force itself out at his mouth. His whole
body was racked and wrenched with an insupportable anguish! But his
disobedient hands gave no heed to the command. They beat the water
vigorously with quick, downward strokes, forcing him to the surface. He
felt his head emerge; his eyes were blinded by the sunlight; his chest
expanded convulsively, and with a supreme and crowning agony his lungs
engulfed a great draught of air, which instantly he expelled in a
shriek!

He was now in full possession of his physical senses. They were,
indeed, preternaturally keen and alert. Something in the awful
disturbance of his organic system had so exalted and refined them that
they made record of things never before perceived. He felt the ripples
upon his face and heard their separate sounds as they struck. He looked
at the forest on the bank of the stream, saw the individual trees, the
leaves and the veining of each leaf—he saw the very insects upon them:
the locusts, the brilliant bodied flies, the gray spiders stretching
their webs from twig to twig. He noted the prismatic colors in all the
dewdrops upon a million blades of grass. The humming of the gnats that
danced above the eddies of the stream, the beating of the dragon flies’
wings, the strokes of the water spiders’ legs, like oars which had
lifted their boat—all these made audible music. A fish slid along
beneath his eyes and he heard the rush of its body parting the water.

He had come to the surface facing down the stream; in a moment the
visible world seemed to wheel slowly round, himself the pivotal point,
and he saw the bridge, the fort, the soldiers upon the bridge, the
captain, the sergeant, the two privates, his executioners. They were in
silhouette against the blue sky. They shouted and gesticulated,
pointing at him. The captain had drawn his pistol, but did not fire;
the others were unarmed. Their movements were grotesque and horrible,
their forms gigantic.

Suddenly he heard a sharp report and something struck the water smartly
within a few inches of his head, spattering his face with spray. He
heard a second report, and saw one of the sentinels with his rifle at
his shoulder, a light cloud of blue smoke rising from the muzzle. The
man in the water saw the eye of the man on the bridge gazing into his
own through the sights of the rifle. He observed that it was a gray eye
and remembered having read that gray eyes were keenest, and that all
famous marksmen had them. Nevertheless, this one had missed.

A counter-swirl had caught Farquhar and turned him half round; he was
again looking at the forest on the bank opposite the fort. The sound of
a clear, high voice in a monotonous singsong now rang out behind him
and came across the water with a distinctness that pierced and subdued
all other sounds, even the beating of the ripples in his ears. Although
no soldier, he had frequented camps enough to know the dread
significance of that deliberate, drawling, aspirated chant; the
lieutenant on shore was taking a part in the morning’s work. How coldly
and pitilessly—with what an even, calm intonation, presaging, and
enforcing tranquility in the men—with what accurately measured interval
fell those cruel words:

“Company!… Attention!… Shoulder arms!… Ready!… Aim!… Fire!”

Farquhar dived—dived as deeply as he could. The water roared in his
ears like the voice of Niagara, yet he heard the dull thunder of the
volley and, rising again toward the surface, met shining bits of metal,
singularly flattened, oscillating slowly downward. Some of them touched
him on the face and hands, then fell away, continuing their descent.
One lodged between his collar and neck; it was uncomfortably warm and
he snatched it out.

As he rose to the surface, gasping for breath, he saw that he had been
a long time under water; he was perceptibly farther downstream—nearer
to safety. The soldiers had almost finished reloading; the metal
ramrods flashed all at once in the sunshine as they were drawn from the
barrels, turned in the air, and thrust into their sockets. The two
sentinels fired again, independently and ineffectually.

The hunted man saw all this over his shoulder; he was now swimming
vigorously with the current. His brain was as energetic as his arms and
legs; he thought with the rapidity of lightning:

“The officer,” he reasoned, “will not make that martinet’s error a
second time. It is as easy to dodge a volley as a single shot. He has
probably already given the command to fire at will. God help me, I
cannot dodge them all!”

An appalling splash within two yards of him was followed by a loud,
rushing sound, DIMINUENDO, which seemed to travel back through the air
to the fort and died in an explosion which stirred the very river to
its deeps! A rising sheet of water curved over him, fell down upon him,
blinded him, strangled him! The cannon had taken an hand in the game.
As he shook his head free from the commotion of the smitten water he
heard the deflected shot humming through the air ahead, and in an
instant it was cracking and smashing the branches in the forest beyond.

“They will not do that again,” he thought; “the next time they will use
a charge of grape. I must keep my eye upon the gun; the smoke will
apprise me—the report arrives too late; it lags behind the missile.
That is a good gun.”

Suddenly he felt himself whirled round and round—spinning like a top.
The water, the banks, the forests, the now distant bridge, fort and
men, all were commingled and blurred. Objects were represented by their
colors only; circular horizontal streaks of color—that was all he saw.
He had been caught in a vortex and was being whirled on with a velocity
of advance and gyration that made him giddy and sick. In few moments he
was flung upon the gravel at the foot of the left bank of the
stream—the southern bank—and behind a projecting point which concealed
him from his enemies. The sudden arrest of his motion, the abrasion of
one of his hands on the gravel, restored him, and he wept with delight.
He dug his fingers into the sand, threw it over himself in handfuls and
audibly blessed it. It looked like diamonds, rubies, emeralds; he could
think of nothing beautiful which it did not resemble. The trees upon
the bank were giant garden plants; he noted a definite order in their
arrangement, inhaled the fragrance of their blooms. A strange roseate
light shone through the spaces among their trunks and the wind made in
their branches the music of AEolian harps. He had not wish to perfect
his escape—he was content to remain in that enchanting spot until
retaken.

A whiz and a rattle of grapeshot among the branches high above his head
roused him from his dream. The baffled cannoneer had fired him a random
farewell. He sprang to his feet, rushed up the sloping bank, and
plunged into the forest.

All that day he traveled, laying his course by the rounding sun. The
forest seemed interminable; nowhere did he discover a break in it, not
even a woodman’s road. He had not known that he lived in so wild a
region. There was something uncanny in the revelation.

By nightfall he was fatigued, footsore, famished. The thought of his
wife and children urged him on. At last he found a road which led him
in what he knew to be the right direction. It was as wide and straight
as a city street, yet it seemed untraveled. No fields bordered it, no
dwelling anywhere. Not so much as the barking of a dog suggested human
habitation. The black bodies of the trees formed a straight wall on
both sides, terminating on the horizon in a point, like a diagram in a
lesson in perspective. Overhead, as he looked up through this rift in
the wood, shone great golden stars looking unfamiliar and grouped in
strange constellations. He was sure they were arranged in some order
which had a secret and malign significance. The wood on either side was
full of singular noises, among which—once, twice, and again—he
distinctly heard whispers in an unknown tongue.

His neck was in pain and lifting his hand to it found it horribly
swollen. He knew that it had a circle of black where the rope had
bruised it. His eyes felt congested; he could no longer close them. His
tongue was swollen with thirst; he relieved its fever by thrusting it
forward from between his teeth into the cold air. How softly the turf
had carpeted the untraveled avenue—he could no longer feel the roadway
beneath his feet!

Doubtless, despite his suffering, he had fallen asleep while walking,
for now he sees another scene—perhaps he has merely recovered from a
delirium. He stands at the gate of his own home. All is as he left it,
and all bright and beautiful in the morning sunshine. He must have
traveled the entire night. As he pushes open the gate and passes up the
wide white walk, he sees a flutter of female garments; his wife,
looking fresh and cool and sweet, steps down from the veranda to meet
him. At the bottom of the steps she stands waiting, with a smile of
ineffable joy, an attitude of matchless grace and dignity. Ah, how
beautiful she is! He springs forwards with extended arms. As he is
about to clasp her he feels a stunning blow upon the back of the neck;
a blinding white light blazes all about him with a sound like the shock
of a cannon—then all is darkness and silence!

Peyton Farquhar was dead; his body, with a broken neck, swung gently
from side to side beneath the timbers of the Owl Creek bridge."""
