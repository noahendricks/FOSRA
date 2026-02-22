from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from backend.src.domain.enums import EmbedderType

if TYPE_CHECKING:
    pass


class EmbedderService:
    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    # get langchain embedder based on enum

    # initial support: ollama,fastembed, huggingface

    # from langchain_ollama import OllamaEmbeddings
    # from langchain_huggingface import HuggingFaceEmbeddings
    # from langchain_community.embeddings import fastembed  

    # embed chunks
        # init chunk list

        # for each chunk: call embed_documents on chunk text
    
        # return chunks list with embeddings

    # embed query(s)
        # call embed_query
    
        # return embedding 

    # embed summary
        # call embed_documents on doc summary

        # return embedding

