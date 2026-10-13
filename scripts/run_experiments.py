"""Train the four conditions with three seeds each, then evaluate and summarize.

Extra arguments are passed to every training run, e.g. ``--num-workers 8``.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

train = [sys.executable, "-m", "food_classifier.train", *sys.argv[1:]]
commands = [
    [*train, "--seed", str(seed), *pretrained, *augment]
    for seed in (0, 1, 2)
    for pretrained in ([], ["--pretrained"])
    for augment in ([], ["--augment"])
]
commands += [
    [sys.executable, "-m", "food_classifier.evaluate"],
    [sys.executable, "-m", "food_classifier.report"],
]

for command in commands:
    print("+", " ".join(command[1:]), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
