FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small; inference does not need CUDA.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV MODEL_PATH=/app/models/resnet18_food12.pt
EXPOSE 8000
CMD ["uvicorn", "food_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
