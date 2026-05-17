import os
import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pytorch3dunet.unet3d.model import UNet3D

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HDF5_PATH  = "cool_dataset_test.h5"
CHECKPOINT = "./checkpoints/best_unet_10000.pth"

TEMP_RANGE = 353.15 - 293.15  # 60°C — adjust if UNet uses different normalisation
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class UNet3DDataset(Dataset):
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.h5file    = h5py.File(hdf5_path, "r")
        self.raw       = self.h5file["raw"]
        self.label     = self.h5file["label"]

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.raw[idx],   dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.float32)
        return {"raw": x, "label": y}


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model():
    model = UNet3D(
        in_channels=3,
        out_channels=1,
        final_sigmoid=False,
        is_segmentation=False,
        f_maps=64,
        num_levels=3
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {CHECKPOINT}")
    return model


# ─────────────────────────────────────────────
# Error field visualisation
# ─────────────────────────────────────────────
def visualise_error_field_3d(y_true, y_pred, sample_idx, threshold_C=10.0):
    # squeeze channel dim: (1, 1, Z, Y, X) → (Z, Y, X)
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    Z   = y_true.shape[0]   # now first dim is Z
    mid = Z // 2

    true_slice  = y_true[mid] * TEMP_RANGE + 293.15   # (Y, X)
    pred_slice  = y_pred[mid] * TEMP_RANGE + 293.15
    error_slice = np.abs(pred_slice - true_slice)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"Sample {sample_idx} | Z-slice {mid} | "
        f"MaxErr={error_slice.max():.1f}°C | MeanErr={error_slice.mean():.2f}°C",
        fontsize=11
    )

    im0 = axes[0].imshow(true_slice, cmap="hot", origin="lower")
    plt.colorbar(im0, ax=axes[0], label="Temperature (K)")
    axes[0].set_title("Ground Truth Temperature")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    im1 = axes[1].imshow(pred_slice, cmap="hot", origin="lower")
    plt.colorbar(im1, ax=axes[1], label="Temperature (K)")
    axes[1].set_title("Predicted Temperature")
    axes[1].set_xlabel("x")

    im2 = axes[2].imshow(error_slice, cmap="RdYlGn_r", origin="lower",
                         vmin=0, vmax=max(10, error_slice.max()))
    plt.colorbar(im2, ax=axes[2], label="Abs Error (°C)")
    axes[2].set_title(f"Prediction Error | red = >{threshold_C}°C")
    axes[2].set_xlabel("x")

    bad_y, bad_x = np.where(error_slice > threshold_C)
    if len(bad_y) > 0:
        axes[2].scatter(bad_x, bad_y, c="red", s=5, zorder=5,
                        label=f">{threshold_C}°C ({len(bad_y)} voxels)")
        axes[2].legend(fontsize=8)

    os.makedirs("error_field_plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"error_field_plots/unet_error_{sample_idx:02d}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error_field_plots/unet_error_{sample_idx:02d}.png")


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader):
    criterion         = nn.MSELoss()
    total_loss        = 0.0
    max_error         = 0.0
    per_sample_errors = []
    all_voxel_errors  = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["raw"].to(DEVICE)
            y = batch["label"].to(DEVICE)   

            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

            # per-voxel errors in °C
            voxel_errors = (pred - y).abs() * TEMP_RANGE   # (B, 1, Z, Y, X)

            # iterate over batch dimension
            for b in range(x.shape[0]):
                sample_errors = voxel_errors[b].cpu().flatten()
                err           = sample_errors.max().item()
                max_error     = max(max_error, err)
                per_sample_errors.append(err)
                all_voxel_errors.append(sample_errors)

                # visualise each sample
                y_true_np = y[b:b+1].cpu().numpy()
                y_pred_np = pred[b:b+1].cpu().numpy()
                sample_idx = i * loader.batch_size + b
                visualise_error_field_3d(y_true_np, y_pred_np, sample_idx,
                                         threshold_C=10.0)

                print(f"Sample {sample_idx:2d} | Loss {loss.item():.6f} | "
                      f"MaxErr {err:.2f}°C")

    # ── Per-sample summary ────────────────────────────────────
    avg_loss = total_loss / len(loader)

    print("\n===== PER-SAMPLE RESULTS =====")
    print(f"Samples        : {len(per_sample_errors)}")
    print(f"Avg MSE Loss   : {avg_loss:.6f}")
    print(f"Worst Error    : {max_error:.2f}°C")

    # ── Per-voxel distribution ────────────────────────────────
    all_voxel_errors = torch.cat(all_voxel_errors)
    total_voxels     = len(all_voxel_errors)

    print("\n===== PER-VOXEL ERROR DISTRIBUTION =====")
    print(f"Total voxels evaluated  : {total_voxels:,}")
    print(f"Mean abs error          : {all_voxel_errors.mean():.2f}°C")
    print(f"Median error            : {all_voxel_errors.median():.2f}°C")
    print(f"90th percentile         : {all_voxel_errors.quantile(0.90):.2f}°C")
    print(f"95th percentile         : {all_voxel_errors.quantile(0.95):.2f}°C")
    print(f"99th percentile         : {all_voxel_errors.quantile(0.99):.2f}°C")
    print(f"99.9th percentile       : {all_voxel_errors.quantile(0.999):.2f}°C")
    print(f"Max error               : {all_voxel_errors.max():.2f}°C")
    print(f"Voxels > 5°C error      : {(all_voxel_errors>5).sum():,}  ({(all_voxel_errors>5).float().mean()*100:.3f}%)")
    print(f"Voxels > 10°C error     : {(all_voxel_errors>10).sum():,}  ({(all_voxel_errors>10).float().mean()*100:.3f}%)")
    print(f"Voxels > 20°C error     : {(all_voxel_errors>20).sum():,}  ({(all_voxel_errors>20).float().mean()*100:.3f}%)")
    print(f"Voxels > 30°C error     : {(all_voxel_errors>30).sum():,}  ({(all_voxel_errors>30).float().mean()*100:.3f}%)")


    # ── Error histogram ───────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.hist(all_voxel_errors.numpy(), bins=100, color="#e76f51",
             edgecolor="none", alpha=0.8, log=True)
    plt.xlabel("Absolute error (°C)")
    plt.ylabel("Voxel count (log scale)")
    plt.title(f"Per-voxel error distribution — {total_voxels:,} voxels")
    plt.axvline(all_voxel_errors.mean().item(), color="red", ls="--",
                label=f"Mean {all_voxel_errors.mean():.1f}°C")
    plt.axvline(all_voxel_errors.quantile(0.99).item(), color="orange", ls="--",
                label=f"99th pct {all_voxel_errors.quantile(0.99):.1f}°C")
    plt.legend()
    plt.tight_layout()
    plt.savefig("unet_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: unet_error_distribution.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset = UNet3DDataset(HDF5_PATH)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"Loaded {len(dataset)} test samples")
    model = load_model()
    model.to(DEVICE)
    evaluate(model, loader)