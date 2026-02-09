from fastapi import APIRouter


router = APIRouter(prefix="/search", tags=["Search"])


# # POST Search across conversations
# @router.post("/search")
# async def search_conversations(request):
#     pass
#
#
# # POST Search within specific conversation
# @router.post("/conversation/{convo_id}/search")
# async def search_in_conversation(convo_id: str, request):
#     pass
