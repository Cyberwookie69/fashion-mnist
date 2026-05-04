# Project flow: fashion_mnist.py and matrix_sweep.py

Two scripts, one assignment, one shared dataset. The assignment-mapping
table shows which Part of the brief each function answers; the flowcharts
show data and asset flow.

## Assignment mapping

| Assignment item | Script | Function |
|---|---|---|
| Part 1: Read data, train/test split, view images | `fashion_mnist.py` | `load_split`, `preview_samples` |
| Part 2: Sigmoid + SGD + cross-entropy, E=10/B=1000 | `fashion_mnist.py` | `build_fc('sigmoid')` + `run_fc_multi_seed` |
| Part 3a: Replace sigmoid with ReLU | `fashion_mnist.py` | `build_fc('relu')` + `run_fc_multi_seed` |
| Part 3b: Add Dropout(0.3) | `fashion_mnist.py` | `build_fc('relu', dropout=0.3)` + `run_fc_multi_seed` |
| Modern smooth activations | `fashion_mnist.py` | `build_fc('gelu')`, `build_fc('mish')` |
| Learning-rate sensitivity (sigmoid + ReLU, 5 seeds) | `fashion_mnist.py` | `run_lr_sweep` |
| Classical baselines | `fashion_mnist.py` | `run_classical_baselines` |
| Multi-seed top-5 sanity check | `fashion_mnist.py` | `run_multi_seed_check` (reads `matrix_results.csv`) |
| Headline seed variance (5 setups x 5 seeds) | `fashion_mnist.py` | `headline_seed_variance` |
| Gradient-flow diagnostic | `fashion_mnist.py` | `GradNormCallback` + `plot_grad_norms_overlay` |
| 3D wireframe overlay (sigmoid vs ReLU) | `fashion_mnist.py` | `plot_sigmoid_relu_3d` (reads `matrix_results.csv`) |
| 1300-cell (E x B x model) grid sweep | `matrix_sweep.py` | `run_sweep`, `analyze` |

## fashion_mnist.py

Vertical execution order (matches `main()` line by line). Outputs branch
off to the right of each step.

```mermaid
flowchart TD
    INPUT[(archive 1<br/>4 IDX/ubyte files)]
    LOAD[load_split<br/>normalize to 0..1]
    PREVIEW[Part 1: preview_samples]
    FC[Parts 2-4: 5 FC variants x 5 seeds<br/>Sigmoid, ReLU, ReLU+Drop0.3,<br/>GELU, Mish]
    PLOT_TC[plot_training_curves<br/>5 series, +/- 1 std band]
    PLOT_GN[plot_grad_norms_overlay<br/>9 lines, log y]
    PCC[write_per_class_csv]
    LRS[Part 6: run_lr_sweep<br/>5 LR x 2 act x 5 seeds]
    CLA[Part 7: run_classical_baselines<br/>LogReg + RandomForest]
    MSC[Part 8: run_multi_seed_check<br/>top-5 cells x 3 seeds]
    HSV[headline_seed_variance<br/>5 setups x 5 seeds]
    GRA[gradient_ratios_table]
    P3D[plot_sigmoid_relu_3d<br/>red sigmoid + blue ReLU wireframe]
    REPORT[write_summary]

    OUT_PREV([samples_preview.png])
    OUT_TC([training_curves.png])
    OUT_GN([gradient_norms_per_layer.png])
    OUT_PCC([per_class_accuracy.csv])
    OUT_LR([lr_sweep_results.csv])
    OUT_CLASS([classical_baselines.csv])
    OUT_SEED([seed_check_results.csv])
    OUT_HEAD([headline_extras.csv])
    OUT_3D([sigmoid_relu_3d.png])
    OUT_REPORT([results_summary.md])

    MATRIX_CSV[(matrix_results.csv<br/>from matrix_sweep.py)]

    INPUT --> LOAD
    LOAD --> PREVIEW --> OUT_PREV
    PREVIEW --> FC
    FC --> PLOT_TC --> OUT_TC
    PLOT_TC --> PLOT_GN --> OUT_GN
    PLOT_GN --> PCC --> OUT_PCC
    PCC --> LRS --> OUT_LR
    LRS --> CLA --> OUT_CLASS
    CLA --> MSC --> OUT_SEED
    MATRIX_CSV --> MSC
    MSC --> HSV --> OUT_HEAD
    HSV --> GRA
    GRA --> P3D --> OUT_3D
    MATRIX_CSV --> P3D
    P3D --> REPORT
    REPORT --> OUT_REPORT

    classDef input fill:#e8f0ff,stroke:#3060c0,color:#000
    classDef out fill:#fff4d6,stroke:#c08020,color:#000
    classDef compute fill:#dff0d8,stroke:#3c763d,color:#000
    classDef plot fill:#f2dede,stroke:#a94442,color:#000
    class INPUT,MATRIX_CSV input
    class OUT_PREV,OUT_TC,OUT_GN,OUT_PCC,OUT_LR,OUT_CLASS,OUT_SEED,OUT_HEAD,OUT_3D,OUT_REPORT out
    class LOAD,PREVIEW,FC,LRS,CLA,MSC,HSV,GRA compute
    class PLOT_TC,PLOT_GN,PCC,P3D,REPORT plot
```

## matrix_sweep.py

```mermaid
flowchart TD
    INPUT[(archive 1<br/>train + test IDX/ubyte)]
    INIT[_worker_init<br/>per-process: load data once,<br/>set TF threading]
    SWEEP[run_sweep<br/>EPOCH_GRID x BATCH_GRID x MODEL_VARIANTS<br/>10 x 10 x 13 = 1300 cells<br/>ProcessPoolExecutor, 6 workers]
    RESUME{cell already<br/>in CSV?}
    TRAIN[_train_cell<br/>build FC, fit, evaluate<br/>jit_compile=True]
    ANALYZE[analyze<br/>load matrix_results.csv,<br/>build heatmap pivots]
    PLOT_DROP[plot_optimum_dropout<br/>per E,B cell: best dropout rate]
    PLOT_RELU[plot_relu_test_acc_heatmap<br/>ReLU only, no dropout]
    SUMMARY[write_summary]

    OUT_CSV([matrix_results.csv<br/>1300 rows])
    OUT_DROP([dropout_optimum_heatmap.png])
    OUT_RELU([matrix_heatmap_test_acc.png])
    OUT_MD([matrix_summary.md])

    INPUT --> INIT --> SWEEP
    SWEEP --> RESUME
    RESUME -- yes, skip --> SWEEP
    RESUME -- no --> TRAIN --> SWEEP
    SWEEP --> OUT_CSV
    OUT_CSV --> ANALYZE
    ANALYZE --> PLOT_DROP --> OUT_DROP
    ANALYZE --> PLOT_RELU --> OUT_RELU
    ANALYZE --> SUMMARY --> OUT_MD

    classDef input fill:#e8f0ff,stroke:#3060c0,color:#000
    classDef out fill:#fff4d6,stroke:#c08020,color:#000
    classDef stage fill:#dff0d8,stroke:#3c763d,color:#000
    classDef decide fill:#f2dede,stroke:#a94442,color:#000
    class INPUT input
    class OUT_CSV,OUT_DROP,OUT_RELU,OUT_MD out
    class INIT,SWEEP,TRAIN,ANALYZE,PLOT_DROP,PLOT_RELU,SUMMARY stage
    class RESUME decide
```

## How the two scripts relate

`matrix_sweep.py` runs once and produces a 1300-row CSV. `fashion_mnist.py`
reads that CSV in two places: Part 8's multi-seed top-5 sanity check (which
retrains the five highest-accuracy cells on extra seeds), and the 3D
wireframe overlay (which pivots the sigmoid and ReLU rows into surfaces).

```mermaid
flowchart LR
    SWEEP[matrix_sweep.py<br/>1300-cell grid sweep<br/>~2 hours one-off]
    MR[(matrix_results.csv<br/>1300 rows)]
    FM[fashion_mnist.py<br/>main pipeline + plotting]

    SWEEP --> MR
    MR --> FM

    classDef stage fill:#dff0d8,stroke:#3c763d,color:#000
    classDef data fill:#e8f0ff,stroke:#3060c0,color:#000
    class SWEEP,FM stage
    class MR data
```

## Render to PNG for Word

Mermaid renders inline on GitHub and in VS Code preview. For an
embedded PNG in the Word document:

```powershell
npm install -g @mermaid-js/mermaid-cli
mmdc -i flowchart.md -o flowchart.png -w 1600 -H 2400
```

Or paste the Mermaid block alone into [mermaid.live](https://mermaid.live)
and export PNG / SVG from there.
