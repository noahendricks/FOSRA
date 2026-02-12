from fastapi import APIRouter


router = APIRouter(tags=["Config"])


# GET User config
async def get_global_config(user_id: str):
    ## get user via id
    #
    # extract global prefs 
    #
    # return global prefs
    pass


# PUT Update user config
# will likely make this per workspace later
@router.put("/user/{user_id}")
async def update_global_config(user_id: str, request):
    # get user via id
    #
    # update global prefs (only changed fields)
    #
    # return global prefs
    pass


# Workspace Config
# GET Workspace config
@router.get("/workspace/{workspace_id}")
async def get_workspace_prefs(workspace_id: str):
    #note: add index on workspace

    # get workspace via id 
    #
    # extract dynamic prefs
    #
    # return dynamic prefs

    pass


# PUT Update workspace config
@router.put("/workspace/{workspace_id}")
async def update_workspace_prefs(workspace_id: str, request):
    #note: patch on workspace dynamic prefs column
    #
    # get workspace via id 
    #
    # update dynamic prefs
    #
    # return dynamic prefs

    pass


# convo config
# GET convo config
@router.get("/workspace/{workspace_id}")
async def get_convo_prefs(workspace_id: str):
    #note: add index on convo

    # get convo via id 
    #
    # extract dynamic prefs
    #
    # return dynamic prefs

    pass


# PUT update convo config
@router.put("/workspace/{workspace_id}")
async def update_convo_prefs(workspace_id: str, request):
    #note: patch on convo dynamic prefs column
    #
    # get convo via id 
    #
    # update dynamic prefs
    #
    # return dynamic prefs

    pass
