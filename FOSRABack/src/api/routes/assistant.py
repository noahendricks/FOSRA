from fastapi import APIRouter

router = APIRouter(prefix="/call", tags=["Calls"])


@router.get("/inference")
async def get_system_config():
    pass


# PUT Update system config
@router.put("/system")
async def update_system_config(request):
    pass
