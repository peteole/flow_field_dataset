import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

from physicsnemo.models.transolver import Transolver
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualise_sample(batch, sample_idx, loss, max_err):
    """
    Extract body positions from input features and plot their layout.
    Body features layout per body: [x, y, z, temp, size_x, size_z]
    All body features are normalised to [0,1] using domain bounds.
    Domain: x=[0,0.5m], y=[0,0.1m], z=[0,0.02m], body_size_max=0.05m
    """
    x = batch["input"]  # (1, N, 40)

    BODY_START    = 3
    MAX_BODIES    = 6
    BODY_FEAT     = 6

    # domain extents in metres — used to convert normalised size to normalised domain coords
    DOMAIN_X      = 0.5
    DOMAIN_Y      = 0.1
    BODY_SIZE_MAX = 0.05

    node0 = x[0, 0, :]

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")
    ax.set_title(f"Sample {sample_idx} | Loss={loss:.5f} | MaxErr={max_err:.1f}°C")

    # draw channel outline
    ax.add_patch(patches.Rectangle(
        (0, 0), 1, 1,
        linewidth=1, edgecolor="black", facecolor="whitesmoke", zorder=0
    ))

    for i in range(MAX_BODIES):
        start = BODY_START + i * BODY_FEAT

        # skip padding slots (all zeros)
        if abs(node0[start:start + BODY_FEAT].sum().item()) < 1e-6:
            continue

        # body position — already normalised to domain x/y
        bx    = node0[start + 0].item()
        by    = node0[start + 1].item()
        btemp = node0[start + 3].item() * 60 + 293.15   

        # size is normalised by BODY_SIZE_MAX
        bsize_x_norm = (node0[start + 4].item() * BODY_SIZE_MAX) / DOMAIN_X
        bsize_y_norm = (node0[start + 5].item() * BODY_SIZE_MAX) / DOMAIN_Y

        is_hot = btemp > 320
        rect = patches.Rectangle(
            (bx - bsize_x_norm / 2, by - bsize_y_norm / 2),
            bsize_x_norm, bsize_y_norm,
            linewidth=1.5,
            edgecolor="darkred"  if is_hot else "navy",
            facecolor="salmon"   if is_hot else "lightblue",
            alpha=0.8,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(
            bx, by, f"{btemp:.0f}K",
            ha="center", va="center", fontsize=7, zorder=3,
            color="darkred" if is_hot else "navy"
        )

    # inflow annotation
    inflow_norm = node0[-1].item()
    inflow_ms   = inflow_norm * (7.0 - 1.0) + 1.0   # denormalise: VEL_MIN=1, VEL_MAX=7
    ax.annotate(
        f"→ {inflow_ms:.1f} m/s",
        xy=(0.02, 0.5), fontsize=9, color="green", zorder=4
    )

    os.makedirs("geometry_plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"geometry_plots/sample_{sample_idx:02d}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved geometry_plots/sample_{sample_idx:02d}.png")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HDF5_PATH  = "cool_dataset_Transolver_test.h5"
CHECKPOINT = "transolver_best.pth"

INPUT_DIM = 40
OUT_DIM   = 1

TEMP_RANGE = 353.15 - 293.15  # 60°C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Dataset (same format!)
# ─────────────────────────────────────────────
class TransolverSurfaceDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, "r") as f:
            self.length = f["coords"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, "_f"):
            self._f = h5py.File(self.hdf5_path, "r")

        n      = int(self._f["n_points"][idx][0])
        coords = torch.tensor(self._f["coords"][idx].reshape(n, 3), dtype=torch.float32)
        x      = torch.tensor(self._f["input"][idx].reshape(n, INPUT_DIM), dtype=torch.float32)
        y      = torch.tensor(self._f["target"][idx].reshape(n, 1), dtype=torch.float32)

        return {"coords": coords, "input": x, "target": y}


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model():
    model = Transolver(
        functional_dim=INPUT_DIM,
        out_dim=OUT_DIM,
        embedding_dim=3,
        n_layers=8,
        n_hidden=128,
        n_head=8,
        slice_num=256,
        structured_shape=None,
        use_te=False
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    # supports both formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded full checkpoint")
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader):
    criterion = nn.MSELoss()

    total_loss = 0.0
    max_error  = 0.0

    per_sample_errors = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            coords = batch["coords"].to(DEVICE)
            x      = batch["input"].to(DEVICE)
            y      = batch["target"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
                pred = model(x, coords)
                loss = criterion(pred, y)

            total_loss += loss.item()

            err = (pred - y).abs().max().item() * TEMP_RANGE
            max_error = max(max_error, err)
            per_sample_errors.append(err)

            # extract metadata from input — inflow is last feature
            inflow_norm = x[0, 0, -1].item()   # normalised inflow
            inflow_ms   = inflow_norm * (7.0 - 1.0) + 1.0  # denormalise

            # max body temp from input features (feature index 3, 9, 15, 21, 27, 33)
            body_temps = [x[0, 0, 3 + i*6].item() * 60 + 293.15 for i in range(6)]
            max_body_temp = max(body_temps)
            visualise_sample(batch, i, loss.item(), err)

            print(f"Step {i:2d} | Loss {loss.item():.6f} | MaxErr {err:.2f}°C | "
                f"Inflow {inflow_ms:.1f}m/s | MaxBodyTemp {max_body_temp:.1f}K")

    avg_loss = total_loss / len(loader)

    print("\n===== TEST RESULTS =====")
    print(f"Samples       : {len(loader)}")
    print(f"Avg MSE Loss  : {avg_loss:.6f}")
    print(f"Worst Error   : {max_error:.2f} °C")
    print(f"Mean Max Err  : {sum(per_sample_errors)/len(per_sample_errors):.2f} °C")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset = TransolverSurfaceDataset(HDF5_PATH)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Loaded {len(dataset)} test samples")

    model = load_model()
    evaluate(model, loader)