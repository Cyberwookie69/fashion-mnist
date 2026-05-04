"""3D wiremesh plot of ReLU test accuracy across (epochs x batch_size).

If matrix_results.csv exists with ReLU rows, uses those. Otherwise runs a
ReLU-only mini-sweep (100 cells, ~3-8 min) and writes relu_sweep.csv.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT_DIR = Path(r"c:\temp\temp\fashion")
MATRIX_CSV = OUT_DIR / "matrix_results.csv"
RELU_CSV = OUT_DIR / "relu_sweep.csv"
OUT = OUT_DIR / "relu_3d.png"


def get_relu_data() -> pd.DataFrame:
    if MATRIX_CSV.exists():
        df = pd.read_csv(MATRIX_CSV)
        sub = df[df["model"] == "ReLU"]
        if len(sub) > 0:
            print(f"using {MATRIX_CSV.name}: {len(sub)} relu cells")
            return sub
    if RELU_CSV.exists():
        sub = pd.read_csv(RELU_CSV)
        print(f"using {RELU_CSV.name}: {len(sub)} relu cells")
        return sub
    print("no cached data; running relu-only sweep (~3-8 min)...")
    from matrix_sweep import (_train_cell, _worker_init,
                              EPOCH_GRID, BATCH_GRID, NUM_WORKERS)
    tasks = [(e, b, "ReLU", "relu", 0.0)
             for e in EPOCH_GRID for b in BATCH_GRID]
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS,
                             initializer=_worker_init) as pool:
        futs = [pool.submit(_train_cell, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if k % 10 == 0:
                print(f"  {k}/{len(tasks)} cells done")
    dt = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(RELU_CSV, index=False)
    print(f"wrote {RELU_CSV.name} ({len(df)} cells, {dt:.1f}s)")
    return df


def main() -> None:
    sub = get_relu_data()
    epoch_grid = sorted(sub["epochs"].unique())
    batch_grid = sorted(sub["batch_size"].unique())
    pivot = sub.pivot_table(index="epochs", columns="batch_size", values="test_acc")
    pivot = pivot.reindex(index=epoch_grid, columns=batch_grid)
    Z = pivot.values
    X_log = np.log10(np.array(batch_grid, dtype=float))
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

    cmap = LinearSegmentedColormap.from_list("lg2k", ["lightgray", "black"])
    norm = Normalize(vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)))
    colors = cmap(norm(np.array(z_mid)))
    lc = Line3DCollection(segs, colors=colors, linewidth=1.2)

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)
    ax.set_xlim(X_log.min() - 0.1, X_log.max() + 0.1)
    ax.set_ylim(min(epoch_grid) - 5, max(epoch_grid) + 5)
    ax.set_zlim(float(np.nanmin(Z)) - 0.01, float(np.nanmax(Z)) + 0.01)
    ax.set_xticks(X_log)
    ax.set_xticklabels([str(b) for b in batch_grid], fontsize=8, rotation=30)
    ax.set_yticks(epoch_grid)
    ax.set_xlabel("batch_size (log)")
    ax.set_ylabel("epochs")
    ax.set_zlabel("test accuracy")
    ax.set_title("ReLU test accuracy over (epochs x batch_size)\n"
                 "darker wire = closer to optimum")
    ax.view_init(elev=28, azim=-58)

    best_i, best_j = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter([X_log[best_j]], [epoch_grid[best_i]], [Z[best_i, best_j]],
               color="red", s=80, marker="*", zorder=10,
               label=f"optimum: E={epoch_grid[best_i]}, "
                     f"B={batch_grid[best_j]}, acc={Z[best_i, best_j]:.4f}")
    ax.legend(loc="upper left", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.1, label="test accuracy")

    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"relu range: {np.nanmin(Z):.4f} .. {np.nanmax(Z):.4f}")
    print(f"optimum at E={epoch_grid[best_i]}, B={batch_grid[best_j]}, "
          f"test_acc={Z[best_i, best_j]:.4f}")


if __name__ == "__main__":
    main()
