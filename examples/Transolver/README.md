# Transolver Training & Evaluation

## Prerequisites

```bash
pip install cooldata physicsnemo h5py torch matplotlib
```

## 1. Prepare the HDF5 Dataset

Edit the config at the top of `prepare_dataset.py`:

| Variable | Description |
|---|---|
| `METADATA_PATH` | Path to `metadata.parquet` |
| `DATA_DIR` | Directory containing `.cgns` files |
| `OUT_PATH` | Output HDF5 file path |
| `NUM_SAMPLES` | Number of samples (`None` = all) |
| `FILTER_VELOCITY` | Velocity range e.g. `(1.0, 7.0)` or `None` |
| `FILTER_N_BODIES` | Exact body count or `None` |

Then run:

```bash
python GenerateTransolverData.py
```

This downloads the required batches and writes `cool_dataset_Transolver.h5`.

## 2. Train

```bash
python train.py
```

Training config at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `HDF5_PATH` | `cool_dataset_Transolver.h5` | Dataset file |
| `EPOCHS` | `200` | Training epochs |
| `LR` | `1e-4` | Learning rate |
| `ACCUMULATION_STEPS` | `4` | Gradient accumulation steps |

The best model is saved to `transolver_best.pth`.

## 3. Evaluate

Point `HDF5_PATH` and `CHECKPOINT` in `test.py` to your test set and checkpoint, then run:

```bash
python test.py
```

Results are printed per sample and summarised at the end:

```
===== TEST RESULTS =====
Samples       : 100
Avg MSE Loss  : 0.000312
Worst Error   : 4.23 °C
Mean Max Err  : 2.11 °C
```

Geometry plots are saved to `geometry_plots/sample_XX.png`.