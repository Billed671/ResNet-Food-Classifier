import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from food_classifier.api import app
from food_classifier.data import CLASSES
from food_classifier.model import build_resnet18


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pt"
    torch.save(
        {"state_dict": build_resnet18(pretrained=False).state_dict(), "classes": CLASSES},
        model_path,
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as client:
        yield client


def test_predict_returns_class_and_confidence(client):
    image = io.BytesIO()
    Image.new("RGB", (320, 240), "orange").save(image, "JPEG")

    response = client.post("/predict", files={"file": ("dish.jpg", image.getvalue(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASSES
    assert 0 < body["confidence"] <= 1


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})

    assert response.status_code == 400
