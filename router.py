from fastapi import APIRouter, File, UploadFile

from classifier import predict

router = APIRouter(
    prefix="/api/classify",
    tags=["classify"],
    responses={404: {"description": "The requested url was not found"}},
)


@router.post("/")
async def classify(file: UploadFile = File(...)):
    prediction = await predict(file)
    return {"prediction": prediction}
