from fastapi import APIRouter, File, UploadFile

router = APIRouter(
    prefix="/api/classify",
    tags=["classify"],
    responses={404: {"description": "The requested page was not found"}},
)


@router.post("/")
async def classify(file: UploadFile = File(...)):
    prediction = await predict(file)
    return {"message": "Hello World"}
