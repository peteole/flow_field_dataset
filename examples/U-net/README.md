# UNet3D Training & Evaluation

## Prerequisites

```bash
pip install cooldata pytorch3dunet h5py torch matplotlib scipy
```

## 1. Prepare the HDF5 Dataset

Edit the config at the top of `GenerateHDF5.py`:

| Variable | Description |
|---|---|
| `METADATA_PATH` | Path to `metadata.parquet` |
| `DATA_DIR` | Directory containing `.cgns` files |
| `OUT_PATH` | Output HDF5 file path |
| `NUM_SAMPLES` | Number of samples (`None` = all) |
| `RESOLUTION` | Voxel grid resolution (default: `64`) |
| `FILTER_VELOCITY` | Velocity range e.g. `(4.0, 7.0)` or `None` |
| `FILTER_N_BODIES` | Exact body count or `None` |

Then run:

```bash
python prepare_dataset.py
```

This writes `cool_dataset.h5` with shape `(N, 3, 64, 64, 64)` for inputs and `(N, 1, 64, 64, 64)` for labels.

Each input has 3 channels:

| Channel | Description |
|---|---|
| 0 | Signed distance field (SDF) |
| 1 | Body temperature map |
| 2 | Inflow velocity (at inlet face) |

## 2. Train

Supports multi-GPU training via PyTorch DDP.

### Single GPU

```bash
python train_unet.py
```

### Multi-GPU

```bash
torchrun --nproc_per_node=NUM_GPUS train_unet.py
```

Training config at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `H5_PATH` | `cool_dataset_unet.h5` | Dataset file |
| `TRAIN_SIZE` | `9000` | Training samples |
| `VAL_SIZE` | `1000` | Validation samples |
| `BATCH_SIZE` | `16` | Batch size per GPU |
| `EPOCHS` | `150` | Training epochs |
| `LR` | `1e-4` | Learning rate |

Checkpoints are saved to `checkpoints/`:

| File | Description |
|---|---|
| `best_unet_<EXP_NAME>.pth` | Best validation loss |
| `last_net_<EXP_NAME>.pth` | Latest epoch |

TensorBoard logs are written to `runs/`:

```bash
tensorboard --logdir runs/
```

## 3. Evaluate

Point `HDF5_PATH` and `CHECKPOINT` in `test.py` to your test set and checkpoint, then run:

```bash
python test.py
```

Results are printed per sample and summarised at the end:

```
===== PER-SAMPLE RESULTS =====
Samples        : 100
Avg MSE Loss   : 0.000312
Worst Error    : 4.23°C

```

Error field slices are saved to `error_field_plots/unet_error_XX.png` and the full voxel error histogram to `unet_error_distribution.png`.