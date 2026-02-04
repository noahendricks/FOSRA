from fastapi import APIRouter

router = APIRouter(prefix="/call", tags=["Calls"])


@router.get("/inference")
async def get_system_config():
    pass


# PUT Update system config
@router.put("/system")
async def update_system_config(request):
    pass


# PUT Update system config
@router.get("/system")
async def call_llm(request):
    # TODO: Figure out streaming implementation
    pass


@router.put("/system")
async def save_assistant_message(
    request,
):
    # TODO: Figure out streaming implementation
    pass
