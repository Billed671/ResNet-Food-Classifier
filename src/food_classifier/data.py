"""Food-101 subset: selected classes, the fixed train/val/test split and image transforms."""

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = (
    "carrot_cake",
    "cheesecake",
    "chocolate_cake",
    "red_velvet_cake",
    "ramen",
    "pho",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "fried_rice",
    "risotto",
    "hamburger",
    "pizza",
)

# Groups of visually similar classes; hamburger and pizza are the distinctive controls.
SIMILAR_GROUPS = (
    ("carrot_cake", "cheesecake", "chocolate_cake", "red_velvet_cake"),
    ("ramen", "pho"),
    ("spaghetti_bolognese", "spaghetti_carbonara"),
    ("fried_rice", "risotto"),
)

TRAIN_PER_CLASS = 600
VAL_PER_CLASS = 150
SPLIT_SEED = 2026

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

train_augmentation = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def make_split(
    official_train: dict[str, list[str]], official_test: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Divide each class's official Food-101 train images into train/val; keep the test images.

    Both arguments map a class name to its image ids, such as ``"ramen/train_001234"``.
    """
    rng = random.Random(SPLIT_SEED)
    split = {"train": [], "val": [], "test": []}
    for name in CLASSES:
        images = sorted(official_train[name])
        rng.shuffle(images)
        split["train"] += images[:TRAIN_PER_CLASS]
        split["val"] += images[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS]
        split["test"] += sorted(official_test[name])
    return split


class Food101Subset(Dataset):
    """One part of the split written by ``prepare_data`` to ``<data_dir>/split.json``."""

    def __init__(self, data_dir: Path, split: str, transform):
        self.images_dir = data_dir / "images"
        self.image_ids = json.loads((data_dir / "split.json").read_text())[split]
        self.labels = [CLASSES.index(image_id.split("/")[0]) for image_id in self.image_ids]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = Image.open(self.images_dir / f"{self.image_ids[index]}.jpg").convert("RGB")
        return self.transform(image), self.labels[index]
