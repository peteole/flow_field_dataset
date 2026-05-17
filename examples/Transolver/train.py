import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
from physicsnemo.models.transolver import Transolver

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
HDF5_PATH          = "cool_dataset_Transolver.h5"
INPUT_DIM          = 40      # Normal(3) + body_feats(36) + inflow(1)
OUT_DIM            = 1       # wall temperature
ACCUMULATION_STEPS = 4       # simulate batch size 4
LR                 = 1e-4
EPOCHS             = 200
TEMP_RANGE         = 353.15 - 293.15   # 60°C — for MaxError conversion

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TransolverSurfaceDataset(Dataset):
    """
    Loads variable-length surface point cloud data from HDF5.
    Each sample has a different N_actual — designed for batch_size=1.

    Returns:
        coords  (N_actual, 3)         — normalised XYZ  → model 'pos'
        input   (N_actual, INPUT_DIM) — physics features → model 'x'
        target  (N_actual, 1)         — normalised wall temperature
    """
    def __init__(self, hdf5_path):
        self.f      = h5py.File(hdf5_path, "r")
        self.length = self.f["coords"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = int(self.f["n_points"][idx][0])
        coords = torch.tensor(
            self.f["coords"][idx].reshape(n, 3), dtype=torch.float32
        )
        x = torch.tensor(
            self.f["input"][idx].reshape(n, INPUT_DIM), dtype=torch.float32
        )
        y = torch.tensor(
            self.f["target"][idx].reshape(n, 1), dtype=torch.float32
        )
        return {"coords": coords, "input": x, "target": y}

    def __del__(self):
        if hasattr(self, "f"):
            self.f.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Dataset & splits ─────────────────────────────────────────────────────
    dataset    = TransolverSurfaceDataset(HDF5_PATH)
    train_size = int(0.8 * len(dataset))
    val_size   = int(0.1 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    torch.manual_seed(32)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"Split: train={train_size}  val={val_size}  test={test_size}")

    # batch_size=1 — variable N_actual per sample
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Transolver(
        functional_dim   = INPUT_DIM,  # 40
        out_dim          = OUT_DIM,    # 1
        embedding_dim    = 3,          # xyz positional embedding
        n_layers         = 8,
        n_hidden         = 256,
        n_head           = 8,
        slice_num        = 256,
        structured_shape = None,
        use_te = False      
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    #criterion     = nn.HuberLoss(delta=1.0)
    criterion = nn.MSELoss()
    scaler        = torch.amp.GradScaler(enabled=(device != "cpu"))
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            coords = batch["coords"].to(device)   # (1, N_actual, 3)
            x      = batch["input"].to(device)    # (1, N_actual, 40)
            y      = batch["target"].to(device)   # (1, N_actual, 1)

            with torch.amp.autocast(device_type=device, enabled=(device != "cpu")):
                pred = model(x, coords)            # (1, N_actual, 1)
                loss = criterion(pred, y) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS

        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_rel_l2  = 0.0
        val_max_err = 0.0

        with torch.no_grad():
            for batch in val_loader:
                coords = batch["coords"].to(device)
                x      = batch["input"].to(device)
                y      = batch["target"].to(device)

                with torch.amp.autocast(device_type=device, enabled=(device != "cpu")):
                    pred = model(x, coords)

                val_loss   += criterion(pred, y).item()
                val_rel_l2 += (torch.norm(pred - y) / torch.norm(y)).item()
                val_max_err = max(
                    val_max_err,
                    (pred - y).abs().max().item() * TEMP_RANGE
                )

        val_loss   /= len(val_loader)
        val_rel_l2 /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"L2Loss={val_rel_l2:.4f}, MaxError={val_max_err:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "transolver_best.pth")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")