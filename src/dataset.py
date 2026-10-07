import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset

class Food101Subset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        img = ImageOps.exif_transpose(img)

        if self.transform:
            img = self.transform(img)

        return img, int(row['label'])