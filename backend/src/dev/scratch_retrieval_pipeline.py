from __future__ import annotations

import asyncio
import mimetypes
import pprint
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from backend.src.domain.enums import VectorStoreType
from backend.src.domain.schemas.config import (
    ChunkerConfig,
    EmbedderConfig,
    UserPreferences,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import (Doc, MDNFile, PDFMetadata, TextMetadata)
from backend.src.services.conversation import llm_service
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.processing.loader_service import (LoaderService, to_bytes)
from backend.src.services.retrieval.vector_service import VectorService

user_dirs = [
    "/home/roccoluxe/Documents/docs/01-ai-ml/ai-sdk/01-core-text-generation",
    "/home/roccoluxe/Documents/docs/01-ai-ml/ai-sdk/00-getting-started",
    "/home/roccoluxe/Documents/docs/01-ai-ml/ai-sdk/02-structured-output",
]

path_dirs = [Path(p) for p in user_dirs]

dir_files: list[MDNFile] = []
for path in path_dirs:
    if path.is_dir():
        files = path.glob("*")
        for file in files:
            if not file.is_dir():
                print(file.as_posix())
                mime = mimetypes.guess_file_type(file.as_posix())
                dir_files.append(
                    MDNFile(
                        type=mime if mime else "text",
                        size=999,
                        name=file.name,
                        bytes=to_bytes(file.as_posix()),
                        media_type=mime if mime else "text",
                    )
                )

cache = Path(".cache")

# docs = LoaderService.parse_files(dir_files)

# pp([d.to_dict() for d in dir_files])

md_bytes = to_bytes(
    "/home/roccoluxe/Documents/docs/09-frontend-ui/tsquery/reference/querying/QueryClient.md"
)

md_blob = Blob.from_data(data=md_bytes)

pdf_bytes = to_bytes("/home/roccoluxe/Documents/Misc/MakingMusic_DennisDeSantis.pdf")

pdf_blob = Blob.from_data(data=pdf_bytes)

mock_mdn_pdf = MDNFile(
    media_type="application/pdf",
    type=pdf_blob.mimetype or "",
    name=str(pdf_blob.path),
    size=0,
    bytes=pdf_bytes,
    webkit_relative_path=pdf_blob.source,
)

mock_mdn_md = MDNFile(
    media_type="text/markdown",
    type=md_blob.mimetype or "",
    name=str(md_blob.path),
    size=0,
    bytes=md_blob.data,
    webkit_relative_path=md_blob.source,
)

result: list[Doc] = LoaderService.parse_files([mock_mdn_md])

from backend.src.services.processing.hi_chunk import HiChunkPipeline

chunker = HiChunkPipeline(config=ChunkerConfig())

out = chunker.index(document=result[0].page_content)


chunks = asyncio.run(
    ChunkerService().chunk_documents(docs=result, config=ChunkerConfig())
)

embedded_chunks = []

for c in chunks:
    embedded_chunks.append(
        asyncio.run(EmbedderService().embed_chunks(chunks=c, config=EmbedderConfig()))
    )

ids = asyncio.run(
    VectorService().upsert(
        config=VectorStoreConfig(),
        chunks=[c for sub in embedded_chunks for c in sub],
        embed_config=EmbedderConfig(),
    )
)


async def print_stream():
    vector_results = await VectorService().search(
        config=VectorStoreConfig(),
        embed_config=EmbedderConfig(),
        query="what is evaluator optimizer?",
    )
    if not vector_results:
        print("no results")
        raise RuntimeError()

    cc = await ConversationService().parse_retrievals(
        retrievals=vector_results, store_type=VectorStoreType.QDRANT
    )

    if not cc:
        raise RuntimeError()

    res = await llm_service.LLMService().generate_llm_response(
        chat_history=[], sources=cc, convo_id="1234", user_prefs=UserPreferences()
    )
    async for chunk in res:
        print(chunk.content, end="", flush=True)


asyncio.run(print_stream())
