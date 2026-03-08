import joblib

from backend.src.domain.schemas.config import ChunkerConfig
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.loader_service import LoaderService

docs = joblib.load(".cache/docs.pkl")

# working on chunker right now
chunks = await ChunkerService().chunk_documents(docs=docs, config=ChunkerConfig())
