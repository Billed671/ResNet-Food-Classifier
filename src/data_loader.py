from torch.utils.data import DataLoader
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.dataset import Food101Subset
from src.transforms import train_tf,eval_tf

train_ds = Food101Subset('data/collected/train.csv', transform=train_tf)
val_ds   = Food101Subset('data/collected/val.csv',   transform=eval_tf)
test_ds  = Food101Subset('data/collected/test.csv',  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)