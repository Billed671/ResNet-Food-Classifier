"""Download Food-101 from Hugging Face and write the selected classes with the fixed split.

Images are saved as ``<data-dir>/images/<class>/<source>_<row>.jpg`` and the split as
``<data-dir>/split.json``. ``source`` is ``train`` or ``test``: the two Hugging Face splits
number their rows independently, so the row index alone does not identify an image.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Image, load_dataset
from tqdm import tqdm

from food_classifier.data import CLASSES, make_split

# Hugging Face publishes Food-101's official test set as the "validation" split.
HF_SPLITS = {"train": "train", "test": "validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset",
        default="ethz/food101",
        help="Hugging Face dataset id or path to a local copy of it",
    )
    args = parser.parse_args()

    sources, official = {}, {}
    for source, hf_split in HF_SPLITS.items():
        # Keep the original JPEG bytes instead of decoding and re-encoding every image.
        dataset = load_dataset(args.dataset, split=hf_split).cast_column(
            "image", Image(decode=False)
        )
        label_names = dataset.features["label"].names
        image_ids = {name: [] for name in CLASSES}
        for row, label in enumerate(dataset["label"]):
            name = label_names[label]
            if name in image_ids:
                image_ids[name].append(f"{name}/{source}_{row:06d}")
        sources[source], official[source] = dataset, image_ids

    split = make_split(official["train"], official["test"])

    for part, image_ids in split.items():
        for image_id in tqdm(image_ids, desc=f"writing {part}"):
            source, row = image_id.split("/")[1].split("_")
            path = args.data_dir / "images" / f"{image_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(sources[source][int(row)]["image"]["bytes"])
    (args.data_dir / "split.json").write_text(json.dumps(split, indent=1))

    for part, image_ids in split.items():
        per_class = Counter(image_id.split("/")[0] for image_id in image_ids)
        print(f"{part}: {len(image_ids)} images, {min(per_class.values())} per class")


if __name__ == "__main__":
    main()
