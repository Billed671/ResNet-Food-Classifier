import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.datasets import Food101
from sklearn.model_selection import train_test_split


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DIR = PROJECT_ROOT / 'data'
COLLECTED_DIR = PROJECT_ROOT / 'data' / 'collected'

DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLASSES = [
    'carrot_cake', 'cheesecake', 'chocolate_cake', 'red_velvet_cake',
    'ramen', 'pho', 'spaghetti_bolognese', 'spaghetti_carbonara',
    'fried_rice', 'risotto', 'hamburger', 'pizza'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

print("Train data is downloading...")
train_ds = Food101(root=str(DIR), split='train', download=True)
print("Test data is downloading...")
test_ds = Food101(root=str(DIR), split='test', download=True)
print("Downloaded!")

def filter_indices(dataset):
    indices, labels = [], []
    for i, (path, label) in enumerate(zip(dataset._image_files, dataset._labels)):
        class_name = dataset.classes[label]
        if class_name in CLASS_TO_IDX:
            indices.append(i)
            labels.append(CLASS_TO_IDX[class_name])
    return indices, labels

train_idx, train_labels = filter_indices(train_ds)
test_idx, test_labels = filter_indices(test_ds)

train_idx, val_idx, train_labels, val_labels = train_test_split(
    train_idx, train_labels,
    test_size=0.2,
    stratify=train_labels,
    random_state=RANDOM_STATE
)
print("Train and Validation data was splitted.")

def make_df(dataset, indices, labels, split):
    rows = []
    for i, y in zip(indices, labels):
        abs_path = Path(dataset._image_files[i]).resolve()
        rel_path = abs_path.relative_to(PROJECT_ROOT)
        rows.append({
            'path': rel_path.as_posix(),
            'label': y,
            'class_name': CLASSES[y],
            'split': split
        })
    return pd.DataFrame(rows)

df_train = make_df(train_ds, train_idx, train_labels, 'train')
df_val   = make_df(train_ds, val_idx,   val_labels,   'val')
df_test  = make_df(test_ds, test_idx,  test_labels,  'test')

df_train.to_csv(COLLECTED_DIR / 'train.csv', index=False)
print(f"Train data was saved in {COLLECTED_DIR / 'train.csv'}")
df_val.to_csv(COLLECTED_DIR / 'val.csv', index=False)
print(f"Validation data was saved in {COLLECTED_DIR / 'val.csv'}")
df_test.to_csv(COLLECTED_DIR / 'test.csv', index=False)
print(f"Test data was saved in {COLLECTED_DIR / 'test.csv'}")