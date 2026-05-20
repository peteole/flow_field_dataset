import os
import time
import platform
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from physicsnemo.models.meshgraphnet import MeshGraphNet
from torch_geometric.data import Data

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR   = "./processed_test"
CHECKPOINT = "./mgn/mgn_best_50000.pth"

NODE_DIM   = 60
EDGE_DIM   = 4
OUT_DIM    = 1
TEMP_RANGE = 353.15 - 293.15
DEVICE     = torch.device("cpu")

HIDDEN_DIM          = 128
NUM_LAYERS_ENCODER  = 2
NUM_LAYERS_DECODER  = 2
NUM_MESSAGE_PASSING = 15
AGGREGATION         = "sum"
BODY_START          = 11   # pos(3) + normal(3) + surf(5)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class MGNDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("sample_*.pt"))
        assert len(self.files) > 0, f"No sample_*.pt files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu",
                          weights_only=False)


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model():
    model = MeshGraphNet(
        input_dim_nodes=NODE_DIM,
        input_dim_edges=EDGE_DIM,
        output_dim=OUT_DIM,
        hidden_dim_node_encoder=HIDDEN_DIM,
        num_layers_node_encoder=NUM_LAYERS_ENCODER,
        hidden_dim_edge_encoder=HIDDEN_DIM,
        num_layers_edge_encoder=NUM_LAYERS_ENCODER,
        hidden_dim_node_decoder=HIDDEN_DIM,
        num_layers_node_decoder=NUM_LAYERS_DECODER,
        hidden_dim_processor=HIDDEN_DIM,
        processor_size=NUM_MESSAGE_PASSING,
        aggregation=AGGREGATION,
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded full checkpoint from {CHECKPOINT}")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded weights from {CHECKPOINT}")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters             : {n_params:,}")
    return model


# ─────────────────────────────────────────────
# Forward helper
# ─────────────────────────────────────────────
def forward_mgn(model, batch, device):
    node_X     = batch["node_X"].to(device)
    edge_X     = batch["edge_X"].to(device)
    node_Y     = batch["node_Y"].to(device)
    edge_index = batch["edges"].to(device)
    graph      = Data(x=node_X, edge_index=edge_index, edge_attr=edge_X)
    pred       = model(node_X, edge_X, graph)
    return pred, node_Y


# ─────────────────────────────────────────────
# Inference time
# ─────────────────────────────────────────────
def measure_inference_time(model, loader, n_runs=100):
    batch      = next(iter(loader))
    node_X     = batch["node_X"].to(DEVICE)
    edge_X     = batch["edge_X"].to(DEVICE)
    edge_index = batch["edges"].to(DEVICE)
    graph      = Data(x=node_X, edge_index=edge_index,
                      edge_attr=edge_X)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(node_X, edge_X, graph)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(node_X, edge_X, graph)
    t1 = time.perf_counter()

    ms = (t1 - t0) / n_runs * 1000
    print(f"Inference time         : {ms:.1f} ms per sample ({n_runs} runs avg)")
    print(f"Inference device       : CPU ({platform.processor()})")
    return ms


# ─────────────────────────────────────────────
# Error field visualisation
# ─────────────────────────────────────────────
def visualise_error_field(batch, pred, node_Y, sample_idx, threshold_C=10.0):
    node_X   = batch["node_X"].cpu().numpy()
    y_true   = node_Y.cpu().numpy().flatten()
    y_pred   = pred.cpu().detach().numpy().flatten()
    errors   = np.abs(y_pred - y_true) * TEMP_RANGE
    y_true_K = y_true * TEMP_RANGE + 293.15

    x_pos = node_X[:, 0]
    y_pos = node_X[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"Sample {sample_idx} | MaxErr={errors.max():.1f}°C | "
        f"MeanErr={errors.mean():.2f}°C", fontsize=12
    )

    sc0 = axes[0].scatter(x_pos, y_pos, c=y_true_K,
                          cmap="hot", s=0.5, alpha=0.6, rasterized=True)
    plt.colorbar(sc0, ax=axes[0], label="Temperature (K)")
    axes[0].set_title("Ground Truth Temperature")
    axes[0].set_xlabel("x (normalised)")
    axes[0].set_ylabel("y (normalised)")

    sc1 = axes[1].scatter(x_pos, y_pos, c=errors,
                          cmap="RdYlGn_r", s=0.5, alpha=0.6,
                          vmin=0, vmax=max(10, errors.max()),
                          rasterized=True)
    plt.colorbar(sc1, ax=axes[1], label="Abs Error (°C)")
    axes[1].set_title(f"Prediction Error | red = >{threshold_C}°C")
    axes[1].set_xlabel("x (normalised)")

    bad = errors > threshold_C
    if bad.sum() > 0:
        axes[1].scatter(x_pos[bad], y_pos[bad], c="red", s=10, zorder=5,
                        label=f">{threshold_C}°C ({bad.sum()} nodes)")
        axes[1].legend(fontsize=8)

    os.makedirs("error_field_plots_mgn", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"error_field_plots_mgn/mgn_error_{sample_idx:02d}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error_field_plots_mgn/mgn_error_{sample_idx:02d}.png")


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader):
    criterion         = nn.MSELoss()
    total_loss        = 0.0
    max_error         = 0.0
    per_sample_errors = []
    all_node_errors   = []
    all_preds         = []
    all_trues         = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            pred, node_Y = forward_mgn(model, batch, DEVICE)
            loss         = criterion(pred, node_Y)
            total_loss  += loss.item()

            node_errors  = (pred - node_Y).abs() * TEMP_RANGE
            err          = node_errors.max().item()
            max_error    = max(max_error, err)
            per_sample_errors.append(err)
            all_node_errors.append(node_errors.cpu().flatten())

            # collect for R²
            all_preds.append(pred.cpu().flatten())
            all_trues.append(node_Y.cpu().flatten())

            inflow_norm   = batch["node_X"][0, -1].item()
            inflow_ms     = inflow_norm * (7.0 - 1.0) + 1.0
            body_temps    = [batch["node_X"][0, BODY_START + j*6 + 3].item() * 60 + 293.15
                             for j in range(6)]
            max_body_temp = max(body_temps)

            visualise_error_field(batch, pred, node_Y, i, threshold_C=10.0)

            print(f"Step {i:2d} | Loss {loss.item():.6f} | MaxErr {err:.2f}°C | "
                  f"Inflow {inflow_ms:.1f}m/s | MaxBodyTemp {max_body_temp:.1f}K")

    avg_loss = total_loss / len(loader)

    print("\n===== PER-SAMPLE RESULTS =====")
    print(f"Samples        : {len(per_sample_errors)}")
    print(f"Avg MSE Loss   : {avg_loss:.6f}")
    print(f"Worst Error    : {max_error:.2f}°C")
    print(f"Mean Max Err   : {np.mean(per_sample_errors):.2f}°C")
    print(f"Median Max Err : {np.median(per_sample_errors):.2f}°C")
    print(f"90th pct Err   : {np.percentile(per_sample_errors, 90):.2f}°C")
    print(f"95th pct Err   : {np.percentile(per_sample_errors, 95):.2f}°C")
    print(f"99th pct Err   : {np.percentile(per_sample_errors, 99):.2f}°C")

    all_node_errors = torch.cat(all_node_errors)
    total_nodes     = len(all_node_errors)

    print("\n===== PER-NODE ERROR DISTRIBUTION =====")
    print(f"Total nodes evaluated  : {total_nodes:,}")
    print(f"Mean abs error         : {all_node_errors.mean():.2f}°C")
    print(f"Median error           : {all_node_errors.median():.2f}°C")
    print(f"90th percentile        : {all_node_errors.quantile(0.90):.2f}°C")
    print(f"95th percentile        : {all_node_errors.quantile(0.95):.2f}°C")
    print(f"99th percentile        : {all_node_errors.quantile(0.99):.2f}°C")
    print(f"99.9th percentile      : {all_node_errors.quantile(0.999):.2f}°C")
    print(f"Max error              : {all_node_errors.max():.2f}°C")
    print(f"Nodes > 5°C error      : {(all_node_errors>5).sum():,}  ({(all_node_errors>5).float().mean()*100:.3f}%)")
    print(f"Nodes > 10°C error     : {(all_node_errors>10).sum():,}  ({(all_node_errors>10).float().mean()*100:.3f}%)")
    print(f"Nodes > 20°C error     : {(all_node_errors>20).sum():,}  ({(all_node_errors>20).float().mean()*100:.3f}%)")
    print(f"Nodes > 30°C error     : {(all_node_errors>30).sum():,}  ({(all_node_errors>30).float().mean()*100:.3f}%)")

    # ── R² score ──────────────────────────────────────────────
    all_preds = torch.cat(all_preds)
    all_trues = torch.cat(all_trues)
    ss_res    = ((all_preds - all_trues) ** 2).sum()
    ss_tot    = ((all_trues - all_trues.mean()) ** 2).sum()
    r2        = (1 - ss_res / ss_tot).item()

    print("\n===== SUMMARY METRICS =====")
    print(f"R² score               : {r2:.4f}")
    print(f"Relative L2 error      : {(all_node_errors / TEMP_RANGE).norm() / (torch.tensor(total_nodes).float().sqrt()):.4f}")

    plt.figure(figsize=(8, 4))
    plt.hist(all_node_errors.numpy(), bins=100, color="#2a9d8f",
             edgecolor="none", alpha=0.8, log=True)
    plt.xlabel("Absolute error (°C)")
    plt.ylabel("Node count (log scale)")
    plt.title(f"MGN per-node error distribution — {total_nodes:,} nodes")
    plt.axvline(all_node_errors.mean().item(), color="red", ls="--",
                label=f"Mean {all_node_errors.mean():.1f}°C")
    plt.axvline(all_node_errors.quantile(0.99).item(), color="orange", ls="--",
                label=f"99th pct {all_node_errors.quantile(0.99):.1f}°C")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mgn_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: mgn_error_distribution.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset = MGNDataset(DATA_DIR)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=lambda x: x[0])
    print(f"Loaded {len(dataset)} test samples from {DATA_DIR}")
    model = load_model()
    #measure_inference_time(model, loader)
    evaluate(model, loader)