"""Standalone: train 3 FC nets at seed 42 and write gradient_norms_per_layer.png.

Trains sigmoid, ReLU, and ReLU+Dropout(0.3) at the assignment hyperparameters
(E=10, B=1000, lr=0.01, seed 42), captures per-Dense-kernel gradient norms
via fashion_mnist.GradNormCallback, and renders a single-panel 9-line overlay.

Roughly 30 seconds total on a multi-threaded 7800X3D, 60-90 seconds on Colab CPU.
Avoids re-running the full fashion_mnist.py pipeline.
"""
from pathlib import Path

from fashion_mnist import (
    load_split, run_fc_experiment, plot_grad_norms_overlay,
    NUM_CLASSES,
)
import tensorflow as tf

OUT_DIR = Path(r"c:\temp\temp\fashion")


def main() -> None:
    print("Loading data...")
    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, NUM_CLASSES).astype("float32")
    y_te_oh = tf.keras.utils.to_categorical(y_te, NUM_CLASSES).astype("float32")

    setups = [
        ("Sigmoid",      "sigmoid", 0.0),
        ("ReLU",         "relu",    0.0),
        ("ReLU+Drop0.3", "relu",    0.3),
    ]
    runs_by_model: dict[str, list[dict]] = {}
    for name, activation, dropout in setups:
        print(f"\nTraining {name}...")
        r = run_fc_experiment(name, activation, dropout,
                              X_tr, y_tr_oh, X_te, y_te_oh,
                              seed=42, lr=0.01, verbose=0)
        tf.keras.backend.clear_session()
        print(f"  test_acc={r['test_acc']:.4f}  ({r['fit_time_s']:.1f}s)")
        runs_by_model[name] = [r]

    print("\nRendering plot...")
    plot_grad_norms_overlay(runs_by_model,
                            ["Sigmoid", "ReLU", "ReLU+Drop0.3"],
                            OUT_DIR / "gradient_norms_per_layer.png",
                            "Per-Dense-kernel gradient norms: sigmoid vs ReLU "
                            "vs ReLU+Drop0.3 (seed 42, log y)")
    print(f"wrote {OUT_DIR / 'gradient_norms_per_layer.png'}")


if __name__ == "__main__":
    main()
