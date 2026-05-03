"""3D wiremesh plot of sigmoid test_acc with epoch grid extended to 1280.

Epochs: [10, 20, 40, 80, 160, 320, 640, 1280] (geometric, x2 each step).
Batches: same as matrix_sweep (32..16384, x2 each step).

8 x 10 = 80 cells. Heavy: E=1280/B=32 takes ~45 min single-process. With
6 parallel workers total wall-clock is roughly 60-90 minutes. Sigmoid only;
no point running ReLU this long since it plateaus far earlier.

Writes sigmoid_extended.csv (resumable: skips cells already in CSV) and
sigmoid_extended_3d.png.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

CSV_OUT = Path(r"c:\temp\temp\fashion\sigmoid_extended.csv")
PNG_OUT = Path(r"c:\temp\temp\fashion\sigmoid_extended_3d.png")

EPOCH_GRID_EXT = [10, 20, 40, 80, 160, 320, 640, 1280]
BATCH_GRID_EXT = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

FIELDS = ["model", "activation", "dropout", "epochs", "batch_size",
          "train_loss", "train_acc", "test_loss", "test_acc", "gap", "fit_time_s"]


def read_done() -> set[tuple[int, int]]:
    if not CSV_OUT.exists():
        return set()
    done: set[tuple[int, int]] = set()
    with open(CSV_OUT, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["epochs"]), int(row["batch_size"])))
            except (KeyError, ValueError):
                continue
    return done


def run_sweep() -> None:
    from matrix_sweep import _train_cell, _worker_init, NUM_WORKERS
    done = read_done()
    target = len(EPOCH_GRID_EXT) * len(BATCH_GRID_EXT)
    print(f"Target {target} cells; {len(done)} already done.")
    tasks = [(e, b, "Sigmoid", "sigmoid", 0.0)
             for e in EPOCH_GRID_EXT for b in BATCH_GRID_EXT
             if (e, b) not in done]
    if not tasks:
        print("Nothing to do.")
        return
    print(f"Submitting {len(tasks)} cells across {NUM_WORKERS} workers...")
    write_mode = "a" if done else "w"
    t0 = time.perf_counter()
    with open(CSV_OUT, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader(); f.flush()
        with ProcessPoolExecutor(max_workers=NUM_WORKERS,
                                 initializer=_worker_init) as pool:
            futs = {pool.submit(_train_cell, t): t for t in tasks}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  ERROR: {type(exc).__name__}: {exc}")
                    continue
                writer.writerow(res); f.flush()
                print(f"  [{k:3d}/{len(tasks)}] E={res['epochs']:4d} "
                      f"B={res['batch_size']:5d} acc={res['test_acc']:.4f} "
                      f"({res['fit_time_s']:.0f}s)")
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min")


def plot() -> None:
    df = pd.read_csv(CSV_OUT)
    epoch_grid = sorted(df["epochs"].unique())
    batch_grid = sorted(df["batch_size"].unique())
    pivot = df.pivot_table(index="epochs", columns="batch_size", values="test_acc")
    pivot = pivot.reindex(index=epoch_grid, columns=batch_grid)
    Z = pivot.values
    X_log = np.log10(np.array(batch_grid, dtype=float))
    Y_log = np.log2(np.array(epoch_grid, dtype=float))
    Xg, Yg = np.meshgrid(X_log, Y_log)

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

    cmap = LinearSegmentedColormap.from_list("lg2k", ["lightgray", "black"])
    norm = Normalize(vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)))
    colors = cmap(norm(np.array(z_mid)))
    lc = Line3DCollection(segs, colors=colors, linewidth=1.2)

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)
    ax.set_xlim(X_log.min() - 0.1, X_log.max() + 0.1)
    ax.set_ylim(Y_log.min() - 0.2, Y_log.max() + 0.2)
    ax.set_zlim(float(np.nanmin(Z)) - 0.01, float(np.nanmax(Z)) + 0.01)
    ax.set_xticks(X_log)
    ax.set_xticklabels([str(b) for b in batch_grid], fontsize=8, rotation=30)
    ax.set_yticks(Y_log)
    ax.set_yticklabels([str(e) for e in epoch_grid], fontsize=8)
    ax.set_xlabel("batch_size (log)")
    ax.set_ylabel("epochs (log)")
    ax.set_zlabel("test accuracy")
    ax.set_title("Sigmoid test accuracy, epochs to 1280\n"
                 "darker wire = closer to optimum")
    ax.view_init(elev=28, azim=-58)

    best_i, best_j = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter([X_log[best_j]], [Y_log[best_i]], [Z[best_i, best_j]],
               color="red", s=80, marker="*", zorder=10,
               label=f"optimum: E={epoch_grid[best_i]}, "
                     f"B={batch_grid[best_j]}, acc={Z[best_i, best_j]:.4f}")
    ax.legend(loc="upper left", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.1, label="test accuracy")

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PNG_OUT}")
    print(f"sigmoid range: {np.nanmin(Z):.4f} .. {np.nanmax(Z):.4f}")
    print(f"optimum at E={epoch_grid[best_i]}, B={batch_grid[best_j]}, "
          f"test_acc={Z[best_i, best_j]:.4f}")


def main() -> None:
    run_sweep()
    plot()


if __name__ == "__main__":
    main()
