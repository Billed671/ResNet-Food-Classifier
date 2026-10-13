# Food Image Classification

[F26] Introduction to Computer Vision, project P05.

ResNet-18 classifier for 12 Food-101 classes, chosen to include groups of visually
similar dishes (cakes, noodles, pasta, rice) alongside two distinctive ones (hamburger,
pizza). The experiments measure how ImageNet pretraining and data augmentation affect
accuracy, macro F1 and confusions between similar classes. The full plan is in
[`docs/project-plan.pdf`](docs/project-plan.pdf).

## Layout

```
src/food_classifier/
  data.py           classes, fixed split, preprocessing and augmentation
  model.py          ResNet-18 with a 12-way head
  prepare_data.py   download Food-101 from Hugging Face, write data/images and data/split.json
  train.py          train one condition with one seed
  evaluate.py       test-set metrics, confusion matrix, misclassified images per run
  report.py         mean/std over seeds, figures, final model export
  api.py            FastAPI prediction service
frontend/           React (Vite) client
scripts/            experiment runner (all conditions and seeds)
tests/              backend tests
docs/               project plan
```

## Setup

Python 3.10+ and Node 20+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[experiments,dev]"
```

The first `pip install` gives the CUDA build of PyTorch; check it with
`python -c "import torch; print(torch.cuda.is_available())"`. On a machine without an
NVIDIA GPU use `--index-url https://download.pytorch.org/whl/cpu` instead.

## Experiments

```bash
python -m food_classifier.prepare_data              # downloads ethz/food101 from Hugging Face
python scripts/run_experiments.py --num-workers 8   # 4 conditions x 3 seeds, evaluate, report
```

`prepare_data` downloads the full Food-101 (about 5 GB, cached by Hugging Face in
`~/.cache/huggingface`) and copies the 12 selected classes to `data/images/`.

Split: 600 train / 150 validation images per class from the official Food-101 training
list (fixed seed), and the official 250 test images per class: 7,200 / 1,800 / 3,000.

| Condition              | Initialization | Augmentation |
|------------------------|----------------|--------------|
| `scratch-plain`        | random         | no (baseline)|
| `pretrained-plain`     | ImageNet       | no           |
| `scratch-augmented`    | random         | yes          |
| `pretrained-augmented` | ImageNet       | yes          |

All runs use full fine-tuning, AdamW (lr 3e-4, weight decay 0.01), batch size 32, 30 epochs
with cosine learning-rate decay, and keep the epoch with the best validation accuracy.
A single run can be trained with, for example:

```bash
python -m food_classifier.train --pretrained --augment --seed 0
```

Each run directory `runs/<condition>/seed<N>/` holds `config.json`, `history.csv`,
`best.pt` and, after evaluation, the test metrics, per-class metrics, confusion matrix and
predictions. The report step writes to `reports/`:

- `summary.csv`: accuracy, macro F1 and share of errors inside similar groups, mean ± std
- `test_metrics.png`: the same as a bar chart
- `per_class_metrics.csv`: precision, recall and F1 per class and condition
- `confusion_pairs.csv`: most frequent confusions per condition
- `confusion_matrix.png`, `misclassified_examples.png`: for the final model
  
### This github repo does not include checkpoint files, as they exceed the 25mb size limit.

The final model is the run with the highest validation accuracy; it is copied to
`models/resnet18_food12.pt`. The test set is used only for reporting.


## Application

```bash
uvicorn food_classifier.api:app --reload          # http://localhost:8000
cd frontend && npm install && npm run dev          # http://localhost:5173
```

`POST /predict` takes a multipart `file` (an image up to 10 MB) and returns
`{"label": "ramen", "confidence": 0.93}`. `MODEL_PATH` overrides the checkpoint location.

With Docker, after `models/resnet18_food12.pt` exists:

```bash
docker compose up --build                          # http://localhost:8080
```

## Tests

```bash
ruff check . && pytest
cd frontend && npm test
```

GitHub Actions runs both, builds the frontend and builds the Docker images on every push
and pull request.

## Team

| Member         | Area                                           |
|----------------|------------------------------------------------|
| Denis Geraskin | AI engineering and backend                     |
| Denis Beliaev  | Data science, data preparation and backend     |
| Aiaz Nadirov   | Infrastructure and deployment                  |
| Marat Yunusov  | Frontend and qualitative analysis              |
