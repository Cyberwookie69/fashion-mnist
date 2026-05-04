"""Dump per-layer kernel gradient L2 norms for sigmoid and ReLU at seed 42.

Trains both at the assignment hyperparameters (Dense(256) -> Dense(128) ->
Dense(10), SGD lr=0.01, ten epochs, batch size 1000), captures per-Dense-
kernel gradient L2 norms per epoch via fashion_mnist.GradNormCallback, and
dumps to CSV.

Output columns: model, seed, epoch, layer_name, n_params, kernel_l2_norm,
rms_per_param. Plus a final printed line with the output/input rms ratio
at epoch 10 for each model.

Run: python dump_grad_norms.py
Time: ~10-15 seconds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from fashion_mnist import load_split, run_fc_experiment, NUM_CLASSES

OUT_DIR = Path(r"c:\temp\temp\fashion")
CSV_OUT = OUT_DIR / "grad_norms_per_epoch.csv"


def main() -> None:
    print("Loading data...")
    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, NUM_CLASSES).astype("float32")
    y_te_oh = tf.keras.utils.to_categorical(y_te, NUM_CLASSES).astype("float32")

    setups = [("Sigmoid", "sigmoid"), ("ReLU", "relu")]
    rows: list[dict] = []
    final_rms: dict[str, dict[str, float]] = {}

    for name, activation in setups:
        print(f"\nTraining {name} (seed=42)...")
        r = run_fc_experiment(name, activation, 0.0,
                              X_tr, y_tr_oh, X_te, y_te_oh,
                              seed=42, lr=0.01, verbose=0)
        tf.keras.backend.clear_session()
        per_epoch = r["layer_grad_norms"]
        sizes = r["kernel_sizes"]
        print(f"  test_acc={r['test_acc']:.4f}, captured {len(per_epoch)} epochs, "
              f"{len(sizes)} kernels")

        # Stable layer order = insertion order in the dict
        layer_keys = list(sizes.keys())
        for epoch_idx, snap in enumerate(per_epoch, start=1):
            for layer_name in layer_keys:
                if layer_name not in snap:
                    continue
                norm = snap[layer_name]
                n_params = sizes[layer_name]
                rms = norm / (n_params ** 0.5) if n_params > 0 else float("nan")
                rows.append({
                    "model": name,
                    "seed": 42,
                    "epoch": epoch_idx,
                    "layer_name": layer_name,
                    "n_params": n_params,
                    "kernel_l2_norm": norm,
                    "rms_per_param": rms,
                })

        # Output/input ratio at epoch 10
        last_snap = per_epoch[-1]
        in_key = layer_keys[0]
        out_key = layer_keys[-1]
        in_rms = last_snap[in_key] / (sizes[in_key] ** 0.5)
        out_rms = last_snap[out_key] / (sizes[out_key] ** 0.5)
        final_rms[name] = {
            "input_layer": in_key, "input_rms": in_rms,
            "output_layer": out_key, "output_rms": out_rms,
            "ratio_output_over_input": out_rms / in_rms if in_rms > 0 else float("nan"),
        }

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nwrote {CSV_OUT} ({len(df)} rows)")

    print("\n=== Output/input rms_per_param ratio at epoch 10 ===")
    for model_name, d in final_rms.items():
        print(f"  {model_name}:")
        print(f"    input  layer = {d['input_layer']:<30s}  "
              f"rms = {d['input_rms']:.6e}")
        print(f"    output layer = {d['output_layer']:<30s}  "
              f"rms = {d['output_rms']:.6e}")
        print(f"    output / input = {d['ratio_output_over_input']:.4f}  "
              f"(input is {1/d['ratio_output_over_input']:.4f}x of output)")


if __name__ == "__main__":
    main()
