"""Read the raw IDX/ubyte files in archive (1)/ and dump a 10-sample preview.

The ubyte files contain the same data as the CSVs, but in MNIST's original
binary encoding (16-byte header for images, 8-byte for labels).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path(r"c:\temp\temp\fashion\archive (1)")
OUT = Path(r"c:\temp\temp\fashion\ubyte_preview.png")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_idx(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic == 2051:
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        if magic == 2049:
            return np.frombuffer(f.read(), dtype=np.uint8)
        raise ValueError(f"unknown magic {magic} in {path}")


def main() -> None:
    X_tr = load_idx(DATA / "train-images-idx3-ubyte")
    y_tr = load_idx(DATA / "train-labels-idx1-ubyte")
    X_te = load_idx(DATA / "t10k-images-idx3-ubyte")
    y_te = load_idx(DATA / "t10k-labels-idx1-ubyte")
    print(f"train: X={X_tr.shape}, y={y_tr.shape}, dtype={X_tr.dtype}")
    print(f"test:  X={X_te.shape}, y={y_te.shape}, dtype={X_te.dtype}")

    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    for i in range(10):
        axes[0, i].imshow(X_tr[i], cmap="gray"); axes[0, i].axis("off")
        axes[0, i].set_title(f"{int(y_tr[i])}: {CLASS_NAMES[int(y_tr[i])]}", fontsize=8)
        axes[1, i].imshow(X_te[i], cmap="gray"); axes[1, i].axis("off")
        axes[1, i].set_title(f"{int(y_te[i])}: {CLASS_NAMES[int(y_te[i])]}", fontsize=8)
    fig.suptitle("Top: train samples 0-9.  Bottom: test samples 0-9.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
