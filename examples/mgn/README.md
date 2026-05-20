# MeshGraphNet Training & Evaluation

## Prerequisites

```bash
pip install cooldata physicsnemo torch-geometric dgl h5py matplotlib
```

## 1. Prepare the Graph Dataset

Edit the config at the top of `GenerateMeshGraphNetData.py`:

| Variable | Description |
|---|---|
| `DATA_PATH` | Path to PyVista dataset directory |
| `CACHE_DIR` | Directory for cached DGL surface graphs |
| `NUM_TOTAL_SAMPLES` | Number of samples to process |
| `OUT_DIR` | Output directory for `.pt` graph files (default: `./processed`) |

Then run:

```bash
python GenerateMeshGraphNetData.py
```

Each sample is saved as `processed/sample_XXXXX.pt` containing:

| Key | Shape | Description |
|---|---|---|
| `node_X` | `(N, 48)` | Node features |
| `edge_X` | `(E, 3)` | Edge features |
| `node_Y` | `(N, 1)` | Target wall temperature |
| `edges` | `(2, E)` | Graph connectivity |
| `sample_id` | `int` | Original sample index |

Node feature layout `(N, 48)`:

| Features | Dim | Description |
|---|---|---|
| Position | 3 | Normalised XYZ cell centres |
| Normal | 3 | Unit surface normal |
| SurfaceType | 5 | One-hot: wall / inlet / outlet / symmetry / body |
| Body features | 36 | 6 heat sources × 6 features, zero-padded |
| Inflow | 1 | Inlet velocity scalar |

## 2. Train

Supports multi-GPU training via PyTorch DDP.

### Single GPU

```bash
python train.py
```

### Multi-GPU

```bash
torchrun --nproc_per_node=NUM_GPUS train.py
```

Training config at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./processed_v2` | Processed graph directory |
| `RUN_SIZES` | `[50000]` | Dataset sizes to train on |
| `EPOCHS` | `200` | Training epochs |
| `LR` | `1e-4` | Learning rate |
| `HIDDEN_DIM` | `128` | Hidden dimension |
| `NUM_MESSAGE_PASSING` | `15` | Message passing steps |

Checkpoints are saved per run size:

| File | Description |
|---|---|
| `mgn_best_<N>.pth` | Best validation loss |
| `checkpoint_<N>.pth` | Latest epoch |

TensorBoard logs are written to `runs/`:

```bash
tensorboard --logdir runs/
```

## 3. Evaluate

Point `DATA_DIR` and `CHECKPOINT` in `test.py` to your test set and checkpoint, then run:

```bash
python test.py
```

Results are printed per sample and summarised at the end: