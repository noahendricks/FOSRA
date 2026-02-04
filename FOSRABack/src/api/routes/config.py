from fastapi import APIRouter

from FOSRABack.src.api.schemas.config_api_schemas import WorkspacePreferencesAPI


router = APIRouter(tags=["Config"])


# GET User config
@router.get("/user/{user_id}")
async def get_user_config(user_id: str) -> WorkspacePreferencesAPI:
    u = WorkspacePreferencesAPI()
    return u


# PUT Update user config
@router.put("/user/{user_id}")
async def update_user_config(user_id: str, request):
    pass


# Workspace Config
# GET Workspace config
@router.get("/workspace/{workspace_id}")
async def get_workspace_config(workspace_id: str):
    pass


# PUT Update workspace config
@router.put("/workspace/{workspace_id}")
async def update_workspace_config(workspace_id: str, request):
    pass


# GET Embedding config
@router.get("/embedding")
async def get_embedding_config():
    pass


# PUT Update embedding config
@router.put("/embedding")
async def update_embedding_config(request):
    pass
