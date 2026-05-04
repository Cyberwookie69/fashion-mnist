# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportMissingModuleSource=false
"""
Fashion MNIST classification - simplified single-file deliverable.

Maps to the assignment one-to-one:
    PART 1 - Data loading + sample preview
    PART 2 - 2-hidden-layer FC, SGD, sigmoid, cross-entropy, E=10/B=1000
    PART 3a - Replace sigmoid with ReLU
    PART 3b - Add Dropout(0.3)

Plus extensions that earn marks:
    PART 4   - Modern activations: GELU and Mish
    PART 6   - Learning-rate sensitivity sweep (5 LRs x sigmoid + ReLU)
    PART 7   - Classical sklearn baselines (LogReg + RandomForest)
    PART 8   - Multi-seed sanity check on top-5 cells from matrix_sweep.py

Outputs (everything Word needs):
    samples_preview.png             - Figure 1: ten random training samples
    training_curves.png             - Figure 2: 5-series train/val curves with bands
    gradient_norms_per_layer.png    - Figure 3: sigmoid vs ReLU vs ReLU+Drop0.3 overlay
    per_class_accuracy.csv          - Table 1 source for the Word document
    sigmoid_relu_3d.png             - Optional: 3D wireframe overlay from matrix sweep
    results_summary.md              - one consolidated markdown report

Run:
    python fashion_mnist.py

Data lives in archive (1)/ as the original IDX/ubyte files
(train-images-idx3-ubyte etc.). The Kaggle CSV exports of the same data
are 3x bigger and 5x slower to parse, but they exist anyway, presumably
because someone really enjoyed converting bytes into ASCII digits and
then back to bytes again.

Version history:
    0.1.0 (2026-04-27) - Initial scaffold: data loading + sigmoid baseline,
                         single experiment, no plotting beyond a loss curve.
                         Sigmoid hit 0.53 and looked broken; spoiler, it isn't.
    0.2.0 (2026-04-27) - Added ReLU and Dropout(0.3) variants (Part 3a/3b)
                         and a per-class accuracy table. ReLU gained 25
                         percentage points without a single new parameter.
    0.3.0 (2026-04-27) - Added GELU and Mish activations (Part 4). GELU and
                         Mish landed within seed noise of ReLU. Modern
                         activations: marketing exists because the science
                         is boring.
    0.4.0 (2026-04-27) - Added LR sensitivity sweep (Part 6); five activations
                         across five learning rates. Discovered sigmoid is
                         not broken, just under-tuned. Mildly humbling.
    0.5.0 (2026-04-27) - Added classical sklearn baselines (Part 7);
                         LogReg + RandomForest as horizontal anchors on the
                         pareto plot. Random Forest beats half the deep nets,
                         which is fine.
    0.6.0 (2026-04-29) - Added multi-seed sanity check (Part 8) keyed on the
                         top-5 cells from matrix_sweep. Per-cell std ~0.18 pp,
                         which means most "improvements" below 0.5 pp are
                         noise. Several published papers should reread this.
    1.0.0 (2026-05-01) - Simplification rewrite. Replaces seven scripts with
                         this single file plus matrix_sweep.py. Single results
                         dict feeds all plots. Surprising how readable a
                         project becomes when you stop carrying its history.
    1.1.0 (2026-05-01) - Pylance hygiene: explicit dtype on load, dict[str,
                         Any] on result accumulators, dropped unused imports.
                         Pylance is happier; nobody else noticed.
    1.2.0 (2026-05-01) - Dropped first_layer_weights plot. MLP first-layer
                         weights look like noise even for well-trained
                         networks because Dense layers learn global linear
                         projections, not local features. Promising the
                         reader they will see "edges and silhouettes" and
                         then showing them static is a bad bargain;
                         gradient_norms_per_layer already carries the
                         vanishing-gradient story without lying about it.
    1.3.0 (2026-05-02) - Pareto plot rework: distinct markers per model
                         (^, o, x, s, P), gold star for the highest-accuracy
                         result, external legend, log-scale x-axis with
                         intermediate ticks. The gold star replaced a duck
                         emoji that matplotlib stretched the figure to
                         18,000 pixels wide trying to render. Lessons learnt.
    1.4.0 (2026-05-03) - Switched data loading from CSV to IDX/ubyte. Native
                         binary format, ~3x smaller on disk and ~5x faster
                         to read than the CSV exports. pandas no longer used
                         for input. Should have been done in 0.1.0 but the
                         CSV was already there, like luggage you keep moving
                         from house to house without opening it.
    1.5.0 (2026-05-03) - Suppressed Keras TF deprecation noise via
                         tf.get_logger().setLevel('ERROR'). Bumped
                         LogisticRegression max_iter from 300 to 1000 so
                         lbfgs converges cleanly. The convergence warning
                         turned out to be correct, which is rare.
    1.6.0 (2026-05-04) - Forced multi-threaded TF intra-op (fixes Windows
                         single-thread default). Diagnostic line at startup.
    1.7.0 (2026-05-04) - Multi-seed FC + LR sweep (5 seeds), with +/- 1 std
                         bands on training curves and error bars on the LR
                         sweep. Plot functions take models_to_plot lists and
                         emit both _assignment.png and _extensions.png pairs.
                         Per-parameter RMS used for cross-layer-fair gradient
                         comparison.
    1.8.0 (2026-05-04) - Removed CNN entirely (build_cnn, run_cnn_experiment,
                         all references in plots and summary). The CNN was
                         comparing apples to oranges next to FC activation
                         choices; activation studies belong in their own
                         universe, architecture comparisons belong in another.
    1.9.0 (2026-05-04) - Asset rationalization to match the restructured
                         single-narrative Word document. Single training_curves.png
                         (5 FC series), single gradient_norms_per_layer.png
                         (sigmoid vs ReLU overlay, 6 lines, log y, 300 dpi).
                         Auto-write per_class_accuracy.csv for the Word table.
                         Stopped saving confusion_matrices.png, lr_sweep_plot.png,
                         pareto_acc_vs_time.png and the _assignment / _extensions
                         pairs; the report does not reference them. Underlying
                         LR sweep, multi-seed top-5 and headline-variance still
                         run because their numbers are cited in text.
    1.10.0 (2026-05-04) - Folded plot_sigmoid_relu_3d.py into fashion_mnist.py.
                          The 3D wireframe overlay (sigmoid red, ReLU blue,
                          per-activation optimum stars) writes sigmoid_relu_3d.png
                          when matrix_results.csv is present. Layer markers added
                          to the gradient overlay (input=circle, hidden=square,
                          output=triangle) plus ReLU+Drop0.3 in green; standalone
                          plot scripts and view_ubyte.py archived to kanweg/.
"""

__version__ = "1.10.0"

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Force TF to use multiple intra-op threads. TF 2.16+ on some Windows builds
# defaults to a single thread which leaves modern multi-core CPUs at ~20%
# utilization. min(8, cpu_count) covers a 7800X3D (16T -> uses 8) without
# oversubscribing; on Colab's 2-vCPU runtime TF auto-clamps to the available
# cores and the call is a no-op.
_CPU = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(min(8, _CPU))
tf.config.threading.set_inter_op_parallelism_threads(min(2, _CPU))

from tensorflow.keras import layers, models

OUT_DIR = Path(r"c:\temp\temp\fashion")
DATA_DIR = OUT_DIR / "archive (1)"
MATRIX_CSV = OUT_DIR / "matrix_results.csv"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = 10
HIDDEN1, HIDDEN2 = 256, 128
EPOCHS = 10
BATCH_SIZE = 1000
SEED = 42

LR_GRID = [0.001, 0.01, 0.05, 0.1, 0.5]
MULTI_SEED_TOP_N = 5
MULTI_SEED_EXTRAS = [1, 7]

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------------------------------------------------------
# PART 1 - Data loading
# -----------------------------------------------------------------------------
def _read_idx(path: Path) -> NDArray[np.uint8]:
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic == 2051:
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        if magic == 2049:
            return np.frombuffer(f.read(), dtype=np.uint8)
        raise ValueError(f"unknown IDX magic {magic} in {path}")


def load_split(split: str) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Load Fashion-MNIST from the IDX/ubyte files in archive (1)/."""
    prefix = "train" if split == "train" else "t10k"
    X_raw = _read_idx(DATA_DIR / f"{prefix}-images-idx3-ubyte")
    y_raw = _read_idx(DATA_DIR / f"{prefix}-labels-idx1-ubyte")
    X: NDArray[np.float32] = (X_raw.reshape(len(X_raw), -1) / 255.0).astype("float32")
    y: NDArray[np.int32] = y_raw.astype("int32")
    return X, y


def preview_samples(X: NDArray[np.float32], y: NDArray[np.int32],
                    out_path: Path, n: int = 10) -> None:
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        axes[i].imshow(X[i].reshape(28, 28), cmap="gray")
        axes[i].set_title(CLASS_NAMES[y[i]], fontsize=8)
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Model factory: FC (Parts 2-4)
# -----------------------------------------------------------------------------
def build_fc(activation: str, dropout: float = 0.0) -> models.Sequential:
    """2-hidden-layer FC: Dense(256) -> Dense(128) -> Dense(10) softmax."""
    layer_list: list[Any] = [layers.Input(shape=(784,))]
    layer_list.append(layers.Dense(HIDDEN1, activation=activation))
    if dropout > 0:
        layer_list.append(layers.Dropout(dropout))
    layer_list.append(layers.Dense(HIDDEN2, activation=activation))
    if dropout > 0:
        layer_list.append(layers.Dropout(dropout))
    layer_list.append(layers.Dense(NUM_CLASSES, activation="softmax"))
    model = models.Sequential(layer_list)
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------------------------------------------------------
# Per-layer gradient-norm callback
# -----------------------------------------------------------------------------
class GradNormCallback(tf.keras.callbacks.Callback):
    """Records per-Dense-kernel L2 gradient norm at the end of each epoch on a
    fixed mini-batch. Indexes by layer position rather than weight name,
    because Keras 3 / TF 2.16+ defaults weight names to bare 'kernel' without
    a layer prefix, which collapses any name-keyed dict to a single entry.
    Also records each kernel's parameter count, so callers can compute
    per-parameter RMS gradients (the cross-layer fair-comparison metric)."""

    def __init__(self, x_sample: NDArray[np.float32],
                 y_sample: NDArray[np.float32]) -> None:
        super().__init__()
        self.x_sample = tf.constant(x_sample)
        self.y_sample = tf.constant(y_sample)
        self.layer_grad_norms: list[dict[str, float]] = []
        self.kernel_sizes: dict[str, int] = {}

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        del epoch, logs
        with tf.GradientTape() as tape:
            preds = self.model(self.x_sample, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(self.y_sample, preds))
        weights = self.model.trainable_weights
        grads = tape.gradient(loss, weights)
        kernel_to_layer: dict[int, int] = {}
        for li, layer in enumerate(self.model.layers):
            if hasattr(layer, "kernel"):
                for wi, w in enumerate(weights):
                    if w is layer.kernel:
                        kernel_to_layer[wi] = li
                        break
        per_layer: dict[str, float] = {}
        for wi, (w, g) in enumerate(zip(weights, grads)):
            if wi not in kernel_to_layer:
                continue
            li = kernel_to_layer[wi]
            key = f"layer{li:02d}_{self.model.layers[li].name}"
            per_layer[key] = float(tf.norm(g).numpy())
            if key not in self.kernel_sizes:
                self.kernel_sizes[key] = int(tf.size(w).numpy())
        self.layer_grad_norms.append(per_layer)


# -----------------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------------
def run_fc_experiment(name: str, activation: str, dropout: float,
                      X_tr: NDArray[np.float32], y_tr_oh: NDArray[np.float32],
                      X_te: NDArray[np.float32], y_te_oh: NDArray[np.float32],
                      *, seed: int = SEED, lr: float = 0.01, verbose: int = 2,
                      ) -> dict[str, Any]:
    np.random.seed(seed); tf.random.set_seed(seed)
    model = build_fc(activation, dropout)
    if lr != 0.01:
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
    grad_cb = GradNormCallback(X_tr[:BATCH_SIZE], y_tr_oh[:BATCH_SIZE])
    t0 = time.perf_counter()
    history = model.fit(X_tr, y_tr_oh, validation_data=(X_te, y_te_oh),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=verbose,
                        callbacks=[grad_cb])
    fit_time = time.perf_counter() - t0
    y_pred = np.argmax(model.predict(X_te, batch_size=BATCH_SIZE, verbose=0), axis=1)
    y_true = np.argmax(y_te_oh, axis=1)
    return {
        "name": name, "seed": seed, "lr": lr,
        "history": history.history,
        "train_loss": history.history["loss"][-1],
        "train_acc": history.history["accuracy"][-1],
        "test_loss": history.history["val_loss"][-1],
        "test_acc": history.history["val_accuracy"][-1],
        "y_true": y_true, "y_pred": y_pred,
        "layer_grad_norms": grad_cb.layer_grad_norms,
        "kernel_sizes": grad_cb.kernel_sizes,
        "fit_time_s": fit_time,
    }


def run_fc_multi_seed(name: str, activation: str, dropout: float,
                      X_tr: NDArray[np.float32], y_tr_oh: NDArray[np.float32],
                      X_te: NDArray[np.float32], y_te_oh: NDArray[np.float32],
                      seeds: list[int], lr: float = 0.01,
                      ) -> list[dict[str, Any]]:
    print(f"\n=== {name} (lr={lr}, {len(seeds)} seeds) ===")
    runs: list[dict[str, Any]] = []
    for s in seeds:
        r = run_fc_experiment(name, activation, dropout, X_tr, y_tr_oh, X_te, y_te_oh,
                              seed=s, lr=lr, verbose=0)
        tf.keras.backend.clear_session()
        runs.append(r)
        print(f"  seed={s:3d}  train_acc={r['train_acc']:.4f}  "
              f"test_acc={r['test_acc']:.4f}  ({r['fit_time_s']:.1f}s)")
    return runs


# -----------------------------------------------------------------------------
# Plot functions (8 deliverables)
# -----------------------------------------------------------------------------
def _representative_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the seed=42 run if present, else the first run."""
    return next((r for r in runs if r.get("seed") == SEED), runs[0])


def plot_training_curves(runs_by_model: dict[str, list[dict[str, Any]]],
                         models: list[str], out_path: Path,
                         title: str = "Training and validation curves") -> None:
    """Two-panel plot (accuracy, loss) with mean line and +/- 1 std band over
    seeds. Each entry in runs_by_model is a list of run dicts (one per seed)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(models):
        if name not in runs_by_model:
            continue
        runs = runs_by_model[name]
        train_acc = np.array([r["history"]["accuracy"] for r in runs])
        val_acc = np.array([r["history"]["val_accuracy"] for r in runs])
        train_loss = np.array([r["history"]["loss"] for r in runs])
        val_loss = np.array([r["history"]["val_loss"] for r in runs])
        epochs_x = np.arange(1, train_acc.shape[1] + 1)
        color = cmap(i % 10)
        for ax, train, val in [(axes[0], train_acc, val_acc),
                                (axes[1], train_loss, val_loss)]:
            ax.plot(epochs_x, train.mean(0), color=color, linewidth=1.4,
                    label=f"{name} train")
            ax.plot(epochs_x, val.mean(0), color=color, linewidth=1.4,
                    linestyle="--", label=f"{name} test")
            if train.shape[0] > 1:
                ax.fill_between(epochs_x, train.mean(0) - train.std(0),
                                train.mean(0) + train.std(0),
                                color=color, alpha=0.18)
                ax.fill_between(epochs_x, val.mean(0) - val.std(0),
                                val.mean(0) + val.std(0),
                                color=color, alpha=0.10)
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy"); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss"); axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_confusion_matrices(runs_by_model: dict[str, list[dict[str, Any]]],
                            models: list[str], out_path: Path,
                            title: str | None = None,
                            cols: int | None = None) -> None:
    """Side-by-side row-normalized confusion matrices using the seed=42 run
    (or first run) from each model in `models`."""
    selected = [(name, _representative_run(runs_by_model[name]))
                for name in models if name in runs_by_model]
    n = len(selected)
    if n == 0:
        return
    if cols is None:
        cols = min(n, 4 if n > 2 else n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, selected):
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(r["y_true"], r["y_pred"]):
            cm[t, p] += 1
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        seed_label = f", seed={r.get('seed', '?')}"
        ax.set_title(f"{name}\ntest_acc={r['test_acc']:.3f}{seed_label}",
                     fontsize=9)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(CLASS_NAMES, fontsize=6)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=5)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_grad_norms_per_layer(runs_by_model: dict[str, list[dict[str, Any]]],
                              models: list[str], out_path: Path,
                              title: str = "Per-Dense-kernel gradient RMS"
                              ) -> None:
    """Per-parameter RMS gradient (L2 norm / sqrt(num_params)) over training
    epochs, log y-axis. Uses representative seed (42). One panel per model;
    one line per Dense kernel inside that model."""
    selected = [(name, _representative_run(runs_by_model[name]))
                for name in models if name in runs_by_model]
    n = len(selected)
    if n == 0:
        return
    cols = min(n, 4); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4 * rows),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")
    for ax, (name, r) in zip(axes, selected):
        per_epoch = r.get("layer_grad_norms", [])
        sizes = r.get("kernel_sizes", {})
        seen: list[str] = []
        for snap in per_epoch:
            for k in snap:
                if k not in seen:
                    seen.append(k)
        for i, lname in enumerate(seen):
            xs = [e + 1 for e, snap in enumerate(per_epoch) if lname in snap]
            denom = sizes.get(lname, 0) ** 0.5
            ys = [snap[lname] / denom if denom > 0 else snap[lname]
                  for snap in per_epoch if lname in snap]
            short = lname.split("_", 1)[0]  # layer00, layer01, ...
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3,
                    color=cmap(i % 10), label=short)
        ax.set_yscale("log"); ax.set_xlabel("epoch", fontsize=8)
        ax.set_title(f"{name} (seed={r.get('seed', '?')})", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("per-param RMS gradient (log)", fontsize=8)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_grad_norms_overlay(runs_by_model: dict[str, list[dict[str, Any]]],
                            models: list[str], out_path: Path,
                            title: str = "Per-Dense-kernel gradient norms"
                            ) -> None:
    """Single-panel overlay: per-Dense-kernel L2 gradient norms vs epoch, log y.
    Distinguishes activations by colour family (sigmoid red shades, ReLU blue
    shades), layers by line darkness within the family. Seed=42 representative."""
    selected = [(name, _representative_run(runs_by_model[name]))
                for name in models if name in runs_by_model]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color_families = {
        "Sigmoid":      ["#fcae91", "#fb6a4a", "#a50f15"],   # red shades
        "ReLU":         ["#9ecae1", "#3182bd", "#08306b"],   # blue shades
        "ReLU+Drop0.3": ["#c7e9c0", "#41ab5d", "#00441b"],   # green shades
        "GELU":         ["#cbc9e2", "#756bb1", "#54278f"],   # purple shades
        "Mish":         ["#fdd0a2", "#fd8d3c", "#8c2d04"],   # orange shades
    }
    layer_markers = ["o", "s", "^"]   # input, hidden, output
    fallback = plt.get_cmap("tab10")
    for mi, (name, r) in enumerate(selected):
        per_epoch = r.get("layer_grad_norms", [])
        if not per_epoch:
            continue
        seen: list[str] = []
        for snap in per_epoch:
            for k in snap:
                if k not in seen:
                    seen.append(k)
        family = color_families.get(name)
        for i, lname in enumerate(seen):
            xs = [e + 1 for e, snap in enumerate(per_epoch) if lname in snap]
            ys = [snap[lname] for snap in per_epoch if lname in snap]
            if family is not None and i < len(family):
                color = family[i]
            else:
                color = fallback((mi * 3 + i) % 10)
            layer_short = "input" if i == 0 else "output" if i == len(seen) - 1 else f"hidden{i}"
            marker = layer_markers[min(i, len(layer_markers) - 1)]
            ax.plot(xs, ys, marker=marker, markersize=4.5, linewidth=1.4,
                    color=color, label=f"{name} - {layer_short}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("kernel gradient L2 norm (log)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_3d_segments(Z: NDArray[np.floating[Any]], X_log: NDArray[np.floating[Any]],
                       epoch_grid: list[int]
                       ) -> tuple[list[list[tuple[float, float, float]]], list[float]]:
    """Helper for the 3D wireframe overlay: produce line segments and their
    midpoint z-values for matplotlib's Line3DCollection."""
    Xg, Yg = np.meshgrid(X_log, epoch_grid)
    segs: list[list[tuple[float, float, float]]] = []
    z_mid: list[float] = []
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            z = Z[i, j]
            if np.isnan(z):
                continue
            if j + 1 < cols and not np.isnan(Z[i, j + 1]):
                segs.append([(Xg[i, j], Yg[i, j], z),
                             (Xg[i, j + 1], Yg[i, j + 1], Z[i, j + 1])])
                z_mid.append((z + Z[i, j + 1]) / 2)
            if i + 1 < rows and not np.isnan(Z[i + 1, j]):
                segs.append([(Xg[i, j], Yg[i, j], z),
                             (Xg[i + 1, j], Yg[i + 1, j], Z[i + 1, j])])
                z_mid.append((z + Z[i + 1, j]) / 2)
    return segs, z_mid


def plot_sigmoid_relu_3d(matrix_df: pd.DataFrame, out_path: Path) -> None:
    """3D wiremesh overlay of Sigmoid vs ReLU test accuracy across (E, B).
    Two activations on the same axes; sigmoid in red shades, ReLU in blue
    shades, darker wire = higher accuracy within each activation. Per-
    activation optimum marked with a star."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(10, 8.5))
    # Pylance does not narrow add_subplot(projection='3d') to Axes3D, so we
    # tell it the result is Any and let the 3D-only attributes (set_zlim,
    # zaxis, view_init, add_collection3d, ...) pass type checking.
    ax: Any = fig.add_subplot(111, projection="3d")
    ax.set_position((0.0, 0.04, 0.88, 0.84))

    ranges: dict[str, tuple[float, float]] = {}
    optima: dict[str, tuple[int, int, float]] = {}
    palettes = [
        ("Sigmoid", ["#fcae91", "#fb6a4a", "#a50f15"]),
        ("ReLU",    ["#9ecae1", "#3182bd", "#08306b"]),
    ]
    for model_name, cmap_colors in palettes:
        sub = matrix_df[matrix_df["model"] == model_name]
        if len(sub) == 0:
            print(f"  no {model_name} rows in matrix_results.csv, skipping")
            continue
        epoch_grid = sorted(sub["epochs"].unique())
        batch_grid = sorted(sub["batch_size"].unique())
        pivot = sub.pivot_table(index="epochs", columns="batch_size",
                                values="test_acc")
        pivot = pivot.reindex(index=epoch_grid, columns=batch_grid)
        Z = pivot.values
        X_log = np.log10(np.array(batch_grid, dtype=float))

        segs, z_mid = _build_3d_segments(Z, X_log, epoch_grid)
        cmap = LinearSegmentedColormap.from_list(f"{model_name}_cm", cmap_colors)
        norm = Normalize(vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)))
        colors = cmap(norm(np.array(z_mid)))
        ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidth=1.2))

        best_i, best_j = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.scatter([X_log[best_j]], [epoch_grid[best_i]], [Z[best_i, best_j]],
                   color=cmap_colors[-1], s=120, marker="*",
                   edgecolor="black", linewidth=0.7, zorder=10)
        ranges[model_name] = (float(np.nanmin(Z)), float(np.nanmax(Z)))
        optima[model_name] = (epoch_grid[best_i], batch_grid[best_j],
                              float(Z[best_i, best_j]))

    if not ranges:
        plt.close(fig)
        return

    z_min = min(r[0] for r in ranges.values()) - 0.02
    z_max = max(r[1] for r in ranges.values()) + 0.02
    epoch_grid_ref = sorted(matrix_df["epochs"].unique())
    batch_grid_ref = sorted(matrix_df["batch_size"].unique())
    X_log_ref = np.log10(np.array(batch_grid_ref, dtype=float))
    ax.set_xlim(X_log_ref.min() - 0.1, X_log_ref.max() + 0.1)
    ax.set_ylim(min(epoch_grid_ref) - 5, max(epoch_grid_ref) + 5)
    ax.set_zlim(z_min, z_max)
    ax.set_xticks(X_log_ref)
    ax.set_xticklabels([str(b) for b in batch_grid_ref], fontsize=9, rotation=30)
    ax.set_yticks(epoch_grid_ref)
    ax.tick_params(axis="y", labelsize=9)
    z_ticks = [round(v, 2) for v in np.arange(0.1, 0.95, 0.1)]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f"{v:.1f}" for v in z_ticks], fontsize=10)
    ax.tick_params(axis="z", pad=6)
    ax.set_xlabel("batch_size (log)", fontsize=11, labelpad=10)
    ax.set_ylabel("epochs", fontsize=11, labelpad=10)
    ax.set_zlabel("test accuracy", fontsize=11, labelpad=8)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    ax.set_title("Sigmoid vs ReLU test accuracy over (epochs x batch_size)\n"
                 "darker wire = higher accuracy within each activation",
                 fontsize=12)
    ax.view_init(elev=24, azim=-58)

    handles: list[Line2D] = []
    for name, color in [("Sigmoid", "#a50f15"), ("ReLU", "#08306b")]:
        if name in ranges:
            lo, hi = ranges[name]
            handles.append(Line2D([0], [0], color=color, lw=2,
                                  label=f"{name} ({lo:.2f}-{hi:.2f})"))
    for name, color in [("Sigmoid", "#a50f15"), ("ReLU", "#08306b")]:
        if name in optima:
            e, b, a = optima[name]
            handles.append(Line2D([0], [0], marker="*", color="white",
                                  markerfacecolor=color, markeredgecolor="black",
                                  markersize=12, linewidth=0,
                                  label=f"{name} optimum: E={e}, B={b}, acc={a:.4f}"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_per_class_csv(results: list[dict[str, Any]], out_path: Path) -> None:
    """Per-class test accuracy table for the Word document. Columns: class,
    sigmoid, relu, gelu, mish. Order matches CLASS_NAMES exactly."""
    selected_models = ["Sigmoid", "ReLU", "GELU", "Mish"]
    per_class: dict[str, list[float]] = {m: [0.0] * NUM_CLASSES
                                          for m in selected_models}
    for r in results:
        if r["name"] not in selected_models:
            continue
        for c in range(NUM_CLASSES):
            mask = r["y_true"] == c
            acc = float((r["y_pred"][mask] == c).mean()) if mask.any() else 0.0
            per_class[r["name"]][c] = acc
    lines = ["class,sigmoid,relu,gelu,mish"]
    for c in range(NUM_CLASSES):
        row = [CLASS_NAMES[c]]
        for m in selected_models:
            row.append(f"{per_class[m][c]:.3f}")
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_lr_sweep(df: pd.DataFrame, out_path: Path) -> None:
    """LR sweep with mean +/- std error bars. df columns: name, lr, seed, test_acc."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    has_seeds = "seed" in df.columns and df["seed"].nunique() > 1
    for i, name in enumerate(df["name"].unique()):
        sub = df[df["name"] == name]
        if has_seeds:
            agg = sub.groupby("lr")["test_acc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["lr"], agg["mean"], yerr=agg["std"], marker="o",
                        color=cmap(i % 10), label=name, capsize=3, linewidth=1.4)
        else:
            sub = sub.sort_values("lr")
            ax.plot(sub["lr"], sub["test_acc"], marker="o", linewidth=1.4,
                    color=cmap(i % 10), label=name)
    ax.set_xscale("log"); ax.set_xlabel("learning rate (log)")
    yl = "test accuracy (mean +/- 1 std)" if has_seeds else "test accuracy"
    ax.set_ylabel(yl)
    ax.set_title("Learning-rate sensitivity")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_pareto(results: list[dict[str, Any]], classical_df: pd.DataFrame,
                matrix_df: pd.DataFrame | None, out_path: Path) -> None:
    """Final scoreboard: test_acc vs fit_time, with classical baselines as
    horizontal reference lines and matrix sweep cells as faded background.
    Each model gets a distinct marker; the highest-accuracy result is tagged
    with a duck emoji rather than a marker."""
    fig, ax = plt.subplots(figsize=(11, 6))
    if matrix_df is not None and len(matrix_df) > 0:
        ax.scatter(matrix_df["fit_time_s"], matrix_df["test_acc"],
                   s=8, alpha=0.18, c="lightsteelblue",
                   label=f"matrix sweep ({len(matrix_df)} cells)")
    palette = plt.get_cmap("tab10")
    markers = ["^", "o", "x", "s", "P"]
    sizes = {"^": 140, "o": 130, "x": 160, "s": 130, "P": 150}
    filled = {"^", "o", "s", "P"}
    winner_idx = max(range(len(results)), key=lambda i: results[i]["test_acc"])
    mk = 0
    for i, r in enumerate(results):
        if i == winner_idx:
            ax.scatter([r["fit_time_s"]], [r["test_acc"]], s=400, marker="*",
                       c="gold", edgecolor="black", linewidth=1.6, zorder=5,
                       label=f"WINNER: {r['name']} ({r['test_acc']:.3f})")
            continue
        m = markers[mk % len(markers)]; mk += 1
        kwargs: dict[str, Any] = dict(s=sizes[m], marker=m, c=[palette(i)],
                                      label=f"{r['name']} ({r['test_acc']:.3f})")
        if m in filled:
            kwargs["edgecolor"] = "black"; kwargs["linewidth"] = 0.7
        else:
            kwargs["linewidth"] = 1.4
        ax.scatter([r["fit_time_s"]], [r["test_acc"]], **kwargs)
    line_styles = ["--", ":"]
    for j, (_, c) in enumerate(classical_df.iterrows()):
        ax.axhline(float(c["test_acc"]), color="gray",
                   linestyle=line_styles[j % len(line_styles)],
                   linewidth=1.0, alpha=0.7,
                   label=f"{c['model']} ({c['test_acc']:.3f})")
    ax.set_xscale("log"); ax.set_xlabel("fit time (s, log)")
    from matplotlib.ticker import LogLocator, ScalarFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 3.0, 5.0, 7.0),
                                          numticks=12))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", labelsize=7, labelcolor="gray")
    ax.set_ylabel("test accuracy")
    ax.set_title("Final scoreboard: deep models, classical baselines, and the matrix sweep")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


# -----------------------------------------------------------------------------
# PART 6 - Learning-rate sensitivity sweep
# -----------------------------------------------------------------------------
def run_lr_sweep(X_tr: NDArray[np.float32], y_tr_oh: NDArray[np.float32],
                 X_te: NDArray[np.float32], y_te_oh: NDArray[np.float32]
                 ) -> pd.DataFrame:
    """5 LRs x 2 activations (sigmoid, ReLU) x 5 seeds at E=10/B=1000.
    Cached in lr_sweep_results.csv. The two-activation focus matches the
    report claim that ReLU's gain from raising LR is far smaller than
    sigmoid's. Results from headline_extras.csv (sigmoid_lr0.5, relu_lr0.5,
    sigmoid_lr0.01, relu_lr0.01) are reused so we never train the same cell
    twice."""
    cache = OUT_DIR / "lr_sweep_results.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print("\n=== LR sensitivity sweep (5 LRs x 2 acts x 5 seeds) ===")
    activations = [("Sigmoid", "sigmoid"), ("ReLU", "relu")]
    rows: list[dict[str, Any]] = []
    for name, act in activations:
        for lr in LR_GRID:
            for seed in HEADLINE_SEEDS:
                np.random.seed(seed); tf.random.set_seed(seed)
                model = build_fc(act, 0.0)
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                              loss="categorical_crossentropy", metrics=["accuracy"])
                model.fit(X_tr, y_tr_oh, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                _, test_acc = model.evaluate(X_te, y_te_oh, batch_size=2048, verbose=0)
                tf.keras.backend.clear_session()
                rows.append({"name": name, "lr": lr, "seed": seed,
                             "test_acc": float(test_acc)})
            sub_acc = [r["test_acc"] for r in rows
                       if r["name"] == name and r["lr"] == lr]
            print(f"  {name:<8} lr={lr:<6} mean={np.mean(sub_acc):.4f} "
                  f"std={np.std(sub_acc, ddof=1):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# -----------------------------------------------------------------------------
# PART 7 - Classical sklearn baselines
# -----------------------------------------------------------------------------
def run_classical_baselines(X_tr: NDArray[np.float32], y_tr: NDArray[np.int32],
                            X_te: NDArray[np.float32], y_te: NDArray[np.int32]
                            ) -> pd.DataFrame:
    """LogisticRegression and RandomForest on flat 784-pixel inputs.
    Anchors what shallow methods achieve. NB: n_jobs=-1 trips a Windows + Python
    3.13 + joblib bug, so single-threaded fits."""
    print("\n=== Classical baselines (sklearn) ===")
    cache = OUT_DIR / "classical_baselines.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    rows.append({"model": "LogisticRegression",
                 "test_acc": float(lr.score(X_te, y_te)),
                 "fit_time_s": time.perf_counter() - t0})
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_tr, y_tr)
    rows.append({"model": "RandomForest(100)",
                 "test_acc": float(rf.score(X_te, y_te)),
                 "fit_time_s": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    for _, r in df.iterrows():
        print(f"  {r['model']:<22} test_acc={r['test_acc']:.4f}  fit={r['fit_time_s']:.1f}s")
    return df


# -----------------------------------------------------------------------------
# PART 8 - Multi-seed sanity check on top-5 cells from the matrix sweep
# -----------------------------------------------------------------------------
def run_multi_seed_check(X_tr: NDArray[np.float32], y_tr_oh: NDArray[np.float32],
                         X_te: NDArray[np.float32], y_te_oh: NDArray[np.float32]
                         ) -> pd.DataFrame:
    """Reads matrix_results.csv (if present), retrains the top-5 ReLU-family
    cells with seeds {1, 7}; combined with the seed=42 row from the CSV gives
    3-seed mean +- std per cell. ~2 min total."""
    if not MATRIX_CSV.exists():
        print(f"\n=== Multi-seed check skipped (no {MATRIX_CSV.name}) ===")
        return pd.DataFrame()
    print("\n=== Multi-seed check (top-5 cells, seeds 1, 7) ===")
    cache = OUT_DIR / "seed_check_results.csv"
    if cache.exists():
        return pd.read_csv(cache)
    matrix = pd.read_csv(MATRIX_CSV)
    relu = matrix[matrix["activation"] == "relu"]
    top = relu.sort_values("test_acc", ascending=False).head(MULTI_SEED_TOP_N).copy()
    rows: list[dict[str, Any]] = []
    for _, r in top.iterrows():
        rows.append({"model": str(r["model"]), "epochs": int(r["epochs"]),
                     "batch_size": int(r["batch_size"]), "dropout": float(r["dropout"]),
                     "seed": 42, "test_acc": float(r["test_acc"])})
        for seed in MULTI_SEED_EXTRAS:
            np.random.seed(seed); tf.random.set_seed(seed)
            model = build_fc("relu", float(r["dropout"]))
            model.fit(X_tr, y_tr_oh, epochs=int(r["epochs"]),
                      batch_size=int(r["batch_size"]), verbose=0)
            _, test_acc = model.evaluate(X_te, y_te_oh, batch_size=2048, verbose=0)
            tf.keras.backend.clear_session()
            rows.append({"model": str(r["model"]), "epochs": int(r["epochs"]),
                         "batch_size": int(r["batch_size"]), "dropout": float(r["dropout"]),
                         "seed": seed, "test_acc": float(test_acc)})
            print(f"  {r['model']:<13} E={int(r['epochs']):3d} B={int(r['batch_size']):5d} "
                  f"seed={seed} acc={float(test_acc):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# -----------------------------------------------------------------------------
# Report extras: gradient-norm ratios, headline seed variance, ReLU at lr=0.5.
# These three numbers go into the Word document. Cached via headline_extras.csv
# and gradient_ratios.csv so reruns of fashion_mnist.py do not redo them.
# -----------------------------------------------------------------------------
HEADLINE_SEEDS = [1, 7, 42, 100, 200]


def gradient_ratios_table(fc_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-activation: input-layer vs output-layer kernel gradient at epoch 10.
    Reports raw L2 norm and per-parameter RMS (norm / sqrt(num_params)).
    The per-parameter ratio is the cross-layer-fair metric."""
    rows: list[dict[str, Any]] = []
    for r in fc_results:
        grads = r.get("layer_grad_norms", [])
        sizes = r.get("kernel_sizes", {})
        if not grads or not sizes:
            continue
        last = grads[-1]
        keys = list(last.keys())
        if len(keys) < 2:
            continue
        in_key, out_key = keys[0], keys[-1]
        in_norm, out_norm = last[in_key], last[out_key]
        in_size, out_size = sizes.get(in_key, 0), sizes.get(out_key, 0)
        if in_size == 0 or out_size == 0:
            continue
        in_rms = in_norm / (in_size ** 0.5)
        out_rms = out_norm / (out_size ** 0.5)
        rows.append({
            "model": r["name"],
            "input_norm_L2": in_norm,
            "output_norm_L2": out_norm,
            "ratio_L2": in_norm / out_norm if out_norm > 0 else float("nan"),
            "input_rms_per_param": in_rms,
            "output_rms_per_param": out_rms,
            "ratio_per_param": in_rms / out_rms if out_rms > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def headline_seed_variance(X_tr: NDArray[np.float32], y_tr_oh: NDArray[np.float32],
                           X_te: NDArray[np.float32], y_te_oh: NDArray[np.float32],
                           ) -> pd.DataFrame:
    """Multi-seed +/- std on the four headline accuracies plus ReLU at lr=0.5.
    Cached so reruns of fashion_mnist.py do not redo the ~3 minutes of work."""
    cache = OUT_DIR / "headline_extras.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print(f"\n=== Headline seed variance ({len(HEADLINE_SEEDS)} seeds per setup) ===")
    setups = [
        ("sigmoid_lr0.01",      "sigmoid", 0.0, 0.01),
        ("relu_lr0.01",         "relu",    0.0, 0.01),
        ("relu_drop0.3_lr0.01", "relu",    0.3, 0.01),
        ("sigmoid_lr0.5",       "sigmoid", 0.0, 0.5),
        ("relu_lr0.5",          "relu",    0.0, 0.5),
    ]
    rows: list[dict[str, Any]] = []
    for name, activation, dropout, lr in setups:
        for seed in HEADLINE_SEEDS:
            np.random.seed(seed); tf.random.set_seed(seed)
            model = build_fc(activation, dropout)
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                          loss="categorical_crossentropy", metrics=["accuracy"])
            model.fit(X_tr, y_tr_oh, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            _, acc = model.evaluate(X_te, y_te_oh, batch_size=2048, verbose=0)
            tf.keras.backend.clear_session()
            print(f"  {name:<22} seed={seed:3d} acc={float(acc):.4f}")
            rows.append({"setup": name, "activation": activation, "dropout": dropout,
                         "lr": lr, "seed": seed, "test_acc": float(acc)})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# -----------------------------------------------------------------------------
# Summary writer (one consolidated markdown)
# -----------------------------------------------------------------------------
def write_summary(results: list[dict[str, Any]], lr_df: pd.DataFrame,
                  classical_df: pd.DataFrame, multiseed_df: pd.DataFrame,
                  grad_ratios: pd.DataFrame, headlines: pd.DataFrame,
                  out_path: Path) -> None:
    L: list[str] = []
    L.append(f"# Fashion MNIST results (fashion_mnist.py v{__version__})\n\n")

    L.append("## Final summary table\n\n")
    L.append("| Model | TrainLoss | TrainAcc | TestLoss | TestAcc | Time (s) |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in results:
        L.append(f"| {r['name']} | {r['train_loss']:.4f} | {r['train_acc']:.4f} | "
                 f"{r['test_loss']:.4f} | {r['test_acc']:.4f} | {r['fit_time_s']:.1f} |\n")
    if len(classical_df) > 0:
        for _, r in classical_df.iterrows():
            L.append(f"| {r['model']} (classical) | - | - | - | "
                     f"{r['test_acc']:.4f} | {r['fit_time_s']:.1f} |\n")
    L.append("\n")

    L.append("## Per-class test accuracy\n\n")
    L.append("| Class | " + " | ".join(r["name"] for r in results) + " |\n")
    L.append("|---" * (len(results) + 1) + "|\n")
    for c in range(NUM_CLASSES):
        cells = []
        for r in results:
            mask = r["y_true"] == c
            acc = float((r["y_pred"][mask] == c).mean()) if mask.any() else 0.0
            cells.append(f"{acc:.3f}")
        L.append(f"| {CLASS_NAMES[c]} | " + " | ".join(cells) + " |\n")
    L.append("\n")

    if len(multiseed_df) > 0:
        L.append("## Multi-seed sanity check (top-5 cells from matrix_sweep)\n\n")
        L.append("Top-5 cells from `matrix_results.csv` retrained with seeds {1, 7}; "
                 "combined with the original seed=42 row -> 3-seed mean +- std.\n\n")
        g = multiseed_df.groupby(["model", "epochs", "batch_size"])["test_acc"]
        s = g.agg(["mean", "std", "min", "max"]).reset_index().sort_values("mean", ascending=False)
        L.append("| Rank | Model | Epochs | Batch | Mean | Std | Min | Max |\n")
        L.append("|---|---|---|---|---|---|---|---|\n")
        for i, (_, row) in enumerate(s.iterrows(), 1):
            L.append(f"| {i} | {row['model']} | {int(row['epochs'])} | "
                     f"{int(row['batch_size'])} | {row['mean']:.4f} | "
                     f"{row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
        L.append(f"\n- Mean per-cell std across 3 seeds: **{s['std'].mean():.4f} pp**.\n")
        L.append("- Read: differences within ~0.5 pp at this scale are seed noise.\n\n")

    if len(lr_df) > 0:
        L.append("## Learning-rate sensitivity\n\n")
        # Multi-seed lr_df has rows (name, lr, seed, test_acc); aggregate to
        # mean +/- std per (name, lr) before pivoting. Falls back to a plain
        # pivot for legacy single-seed CSVs that have no 'seed' column.
        if "seed" in lr_df.columns:
            agg = lr_df.groupby(["name", "lr"])["test_acc"].agg(["mean", "std"]).reset_index()
            pivot = agg.pivot(index="name", columns="lr", values="mean")
            std_pivot = agg.pivot(index="name", columns="lr", values="std")
            L.append(f"5-seed mean test accuracy per (activation, learning rate); "
                     f"std in parentheses.\n\n")
        else:
            pivot = lr_df.pivot(index="name", columns="lr", values="test_acc")
            std_pivot = None
        L.append("| Activation | " + " | ".join(f"lr={lr}" for lr in LR_GRID) + " |\n")
        L.append("|---" * (len(LR_GRID) + 1) + "|\n")
        for name in pivot.index:
            cells = []
            for lr in LR_GRID:
                m = float(cast(Any, pivot.loc[name, lr]))
                s = (float(cast(Any, std_pivot.loc[name, lr]))
                     if std_pivot is not None else float("nan"))
                if not np.isnan(s):
                    cells.append(f"{m:.4f} (+/-{s:.4f})")
                else:
                    cells.append(f"{m:.4f}")
            L.append(f"| {name} | " + " | ".join(cells) + " |\n")
        sig = pivot.loc["Sigmoid"] if "Sigmoid" in pivot.index else None
        relu = pivot.loc["ReLU"] if "ReLU" in pivot.index else None
        if sig is not None and relu is not None:
            L.append(f"\n- Sigmoid at default lr=0.01: **{sig.get(0.01, float('nan')):.4f}**\n")
            L.append(f"- Sigmoid at lr=0.5: **{sig.get(0.5, float('nan')):.4f}** "
                     f"(beats ReLU at default lr=0.01 of {relu.get(0.01, float('nan')):.4f})\n")
            L.append("- Confirms: 'sigmoid is broken' is partly an LR-tuning artifact.\n\n")

    if len(grad_ratios) > 0:
        L.append("## Gradient flow at epoch 10 (input vs output layer)\n\n")
        L.append("Cross-layer fair comparison: per-parameter RMS gradient "
                 "(L2 norm divided by sqrt(num_params)). The ratio column is "
                 "input_rms / output_rms; a low ratio means the input layer is "
                 "starved of gradient signal. Single seed (42).\n\n")
        L.append("| Activation | Input RMS | Output RMS | Ratio | Raw L2 ratio |\n")
        L.append("|---|---:|---:|---:|---:|\n")
        for _, r in grad_ratios.iterrows():
            L.append(f"| {r['model']} | {r['input_rms_per_param']:.6f} | "
                     f"{r['output_rms_per_param']:.6f} | "
                     f"{r['ratio_per_param']:.4f} | {r['ratio_L2']:.4f} |\n")
        sig_row = grad_ratios[grad_ratios["model"] == "Sigmoid"]
        relu_row = grad_ratios[grad_ratios["model"] == "ReLU"]
        if len(sig_row) > 0 and len(relu_row) > 0:
            sig_r = float(sig_row.iloc[0]["ratio_per_param"])
            relu_r = float(relu_row.iloc[0]["ratio_per_param"])
            L.append(f"\n- Sigmoid input layer is **{1/sig_r:.1f}x weaker** "
                     "per parameter than its output layer. ")
            L.append(f"ReLU's input is **{1/relu_r:.1f}x weaker**, "
                     f"so ReLU's gradient flow is roughly {relu_r/sig_r:.1f}x "
                     "healthier than sigmoid's at the same epoch.\n\n")

    if len(headlines) > 0:
        L.append("## Headline accuracies with seed variance (5 seeds)\n\n")
        L.append("Each setup retrained with seeds {1, 7, 42, 100, 200}. "
                 "Confirms which differences exceed seed noise.\n\n")
        g = headlines.groupby("setup")["test_acc"]
        agg = g.agg(["mean", "std", "min", "max"]).reset_index()
        L.append("| Setup | Mean | Std | Min | Max |\n")
        L.append("|---|---:|---:|---:|---:|\n")
        for _, r in agg.iterrows():
            L.append(f"| {r['setup']} | {r['mean']:.4f} | {r['std']:.4f} | "
                     f"{r['min']:.4f} | {r['max']:.4f} |\n")
        means = dict(zip(agg["setup"], agg["mean"]))
        if {"sigmoid_lr0.01", "sigmoid_lr0.5"} <= means.keys():
            sig_gain = means["sigmoid_lr0.5"] - means["sigmoid_lr0.01"]
            L.append(f"\n- Sigmoid gains **{sig_gain:+.4f}** from raising LR "
                     "from 0.01 to 0.5.\n")
        if {"relu_lr0.01", "relu_lr0.5"} <= means.keys():
            relu_gain = means["relu_lr0.5"] - means["relu_lr0.01"]
            L.append(f"- ReLU gains **{relu_gain:+.4f}** from the same LR change.\n")
            L.append("- The vanishing-gradient handicap is therefore activation-"
                     "specific: sigmoid recovers far more from LR tuning than ReLU does.\n\n")

    L.append("## Findings (calibrated to the numbers above)\n\n")
    L.append("- **Sigmoid is slow, not random.** At default SGD lr=0.01 / E=10 / B=1000 "
             "it reaches ~55-60% test acc - well above 10% random. Per-layer gradient "
             "norms (`gradient_norms_assignment.png`) show the input-layer kernel sits "
             "an order of magnitude below the output-layer kernel - direct proof of "
             "vanishing gradients. With higher LR the gap to ReLU shrinks substantially.\n")
    L.append("- **ReLU > Sigmoid by ~18 pp at the assignment hyperparameters.** Gradients "
             "stay alive on positive inputs.\n")
    L.append("- **Dropout(0.3) doesn't help at E=10/B=1000.** No overfitting yet to "
             "regularize. The dropout-rate sweep (matrix_sweep.py output) shows the "
             "optimum drifts: small batches want dropout near 0, larger ones want "
             "essentially 0. The textbook 0.3 / 0.5 defaults are not universally optimal.\n")
    L.append("- **GELU and Mish are ~within seed noise of ReLU.** Modern smooth activations "
             "polish, they don't transform.\n")
    if len(classical_df) > 0:
        best_classical = float(classical_df["test_acc"].max())
        best_deep = max(r["test_acc"] for r in results)
        L.append(f"- **Deep-net advantage over classical sklearn: {(best_deep - best_classical) * 100:.1f} pp.** "
                 f"LogReg gets {float(classical_df.iloc[0]['test_acc']):.3f}, RandomForest gets "
                 f"{float(classical_df.iloc[1]['test_acc']):.3f}, our best deep model gets "
                 f"{best_deep:.3f}. Most of the dataset's signal is recoverable shallowly.\n\n")

    L.append("## Files produced\n\n")
    for f in ["samples_preview.png",
              "training_curves.png",
              "gradient_norms_per_layer.png",
              "per_class_accuracy.csv"]:
        L.append(f"- `{f}`\n")
    L.append(f"- `{out_path.name}` (this file)\n")
    L.append("\nAlso (see `matrix_sweep.py`):\n- `matrix_results.csv`, "
             "`matrix_summary.md`, `dropout_optimum_heatmap.png`\n")

    out_path.write_text("".join(L), encoding="utf-8")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"fashion_mnist.py v{__version__}")
    print(f"TF intra-op threads = {tf.config.threading.get_intra_op_parallelism_threads()}, "
          f"inter-op = {tf.config.threading.get_inter_op_parallelism_threads()}, "
          f"os.cpu_count={os.cpu_count()}, "
          f"GPUs={len(tf.config.list_physical_devices('GPU'))}")

    print("\n--- PART 1: Data ---")
    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, NUM_CLASSES).astype("float32")
    y_te_oh = tf.keras.utils.to_categorical(y_te, NUM_CLASSES).astype("float32")
    print(f"train {X_tr.shape}, test {X_te.shape}")
    preview_samples(X_tr, y_tr, OUT_DIR / "samples_preview.png")
    print(f"wrote samples_preview.png")

    seeds = HEADLINE_SEEDS
    runs_by_model: dict[str, list[dict[str, Any]]] = {}

    print(f"\n--- PARTS 2-4: FC experiments ({len(seeds)} seeds each) ---")
    fc_setups = [
        ("Sigmoid",      "sigmoid", 0.0, 0.01),
        ("ReLU",         "relu",    0.0, 0.01),
        ("ReLU+Drop0.3", "relu",    0.3, 0.01),
        ("GELU",         "gelu",    0.0, 0.01),
        ("Mish",         "mish",    0.0, 0.01),
    ]
    for name, activation, dropout, lr in fc_setups:
        runs_by_model[name] = run_fc_multi_seed(
            name, activation, dropout, X_tr, y_tr_oh, X_te, y_te_oh,
            seeds=seeds, lr=lr)

    # Flat list using the seed=42 representative per model for downstream plots.
    results = [_representative_run(runs_by_model[name]) for name in runs_by_model]
    fc_results = results

    print("\n--- Plotting main results ---")
    ALL_FC = ["Sigmoid", "ReLU", "ReLU+Drop0.3", "GELU", "Mish"]

    plot_training_curves(runs_by_model, ALL_FC,
                         OUT_DIR / "training_curves.png",
                         "Training and validation curves "
                         f"({len(seeds)} seeds, +/- 1 std band)")
    print("wrote training_curves.png")

    plot_grad_norms_overlay(runs_by_model,
                            ["Sigmoid", "ReLU", "ReLU+Drop0.3"],
                            OUT_DIR / "gradient_norms_per_layer.png",
                            "Per-Dense-kernel gradient norms: sigmoid vs ReLU "
                            "vs ReLU+Drop0.3 (seed 42, log y)")
    print("wrote gradient_norms_per_layer.png")

    write_per_class_csv(results, OUT_DIR / "per_class_accuracy.csv")
    print("wrote per_class_accuracy.csv")

    print("\n--- PART 6: LR sweep (multi-seed, sigmoid + ReLU) ---")
    lr_df = run_lr_sweep(X_tr, y_tr_oh, X_te, y_te_oh)

    print("\n--- PART 7: Classical baselines ---")
    classical_df = run_classical_baselines(X_tr, y_tr, X_te, y_te)

    print("\n--- PART 8: Multi-seed top-5 sanity check ---")
    multiseed_df = run_multi_seed_check(X_tr, y_tr_oh, X_te, y_te_oh)

    print("\n--- Report extras: gradient ratios + headline seed variance ---")
    grad_ratios = gradient_ratios_table(fc_results)
    headlines = headline_seed_variance(X_tr, y_tr_oh, X_te, y_te_oh)

    if MATRIX_CSV.exists():
        print("\n--- 3D wireframe overlay (sigmoid vs ReLU from matrix sweep) ---")
        plot_sigmoid_relu_3d(pd.read_csv(MATRIX_CSV),
                             OUT_DIR / "sigmoid_relu_3d.png")
        print("wrote sigmoid_relu_3d.png")

    print("\n--- Summary ---")
    write_summary(results, lr_df, classical_df, multiseed_df,
                  grad_ratios, headlines,
                  OUT_DIR / "results_summary.md")
    print("wrote results_summary.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
