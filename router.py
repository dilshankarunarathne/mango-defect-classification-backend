from fastapi import APIRouter

router = APIRouter(
    prefix="/api/classify",
    tags=["classify"],
    responses={404: {"description": "The requested page was not found"}},
)

