"""HTTP prediction service for the trained food classifier."""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from food_classifier.data import eval_transform
from food_classifier.model import build_resnet18

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("MODEL_PATH", "models/resnet18_food12.pt"))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    app.state.model = model.eval()
    app.state.classes = checkpoint["classes"]
    yield


app = FastAPI(title="Food classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be at most 10 MB")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(400, "File is not a valid image") from None

    with torch.inference_mode():
        probabilities = app.state.model(eval_transform(image).unsqueeze(0)).softmax(1)[0]
    confidence, index = probabilities.max(0)
    return {"label": app.state.classes[index], "confidence": round(confidence.item(), 4)}
