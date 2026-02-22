from __future__ import annotations

from typing import TYPE_CHECKING

from backend.src.domain.schemas.doc import Chunk

if TYPE_CHECKING:
    from backend.src.api.request_context import RequestContext


from backend.src.domain.enums import ChunkerType

class ChunkerService:
    def __init__(
        self,
        default_chunker: ChunkerType = ChunkerType.SEMANTIC,
    ):
        self.default_chunker = default_chunker

    def _get_chunker(self, chunker_type: ChunkerType):
        """Get a chunker instance by type."""
        # TODO: return chonkie chunker based on enum passed
        # pass type enum and get users existing config or use default
        
        # if ChunkerType.LATE : 
            # initial late chunker and dump config kwargs
            # return chunker
        # if ChunkerType.RECURSIVE : 
            # initial recursive chunker and dump config kwargs
            # return chunker
        # if ChunkerType.SEMANTIC : 
            # initial semantic chunker and dump config kwargs
            # return chunker
        # if ChunkerType.NEURAL : 
            # initial recursive chunker and dump config kwargs
            # return chunker
        # ...etc,etc,etc

        return None

    @staticmethod
    def chunk_doc() -> list[Chunk]:

    # bring in parsed doc(s) [page content: str, metadata {"", obj | str | []}]

        # file type check
    
            # if doc is pdf, chunk / (leave as) as pages unless page length > 300 tokens, then chunk w/ text chunker
                # call get_chunker for text based
                # use user preferred chunker and user config for chunker
                # chunk by Document as page, if page > 300 tokens, chunk within the page
    
                # return chunks list
    
            # elif doc is code, chunk using code chunker
                # call get_chunker for code based
                # use user preferred chunker and user config for chunker
                # chunk
    
                # return chunks list

            # elif doc is text file, chunk using text chunker
                # call get_chunker for text based
                # use user preferred chunker and user config for chunker

                # return chunks list
        pass
    
    
    




