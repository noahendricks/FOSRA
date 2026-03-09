from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.services.retrieval.vector_service import VectorService
from backend.src.tasks.processing import embed_query

from .broker import broker


@broker.task
async def vector_source_search():
    #  accept query
    #  --- call reform_query and accept reformed query
    #  --- call embed_query and accept query embedding
    #  --- call vector store on "sources" collection and return top 5 sources and topics
    pass
