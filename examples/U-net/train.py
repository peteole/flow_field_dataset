import os
import h5py
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pytorch3dunet.unet3d.model import UNet3D


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
H5_PATH  = "cool_dataset_unet.h5"
EXP_NAME = "unet_run"

TRAIN_SIZE = 9000
VAL_SIZE   = 1000

BATCH_SIZE = 16
EPOCHS     = 150
LR         = 1e-4
SEED       = 32

CKPT_DIR = "checkpoints"
LOG_DIR  = "logs"
RUN_DIR  = "runs"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(RUN_DIR,  exist_ok=True)


# ─────────────────────────────
# DDP SETUP
# ─────────────────────────────
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device("cuda", local_rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ─────────────────────────────
# DATASET
# ─────────────────────────────
class H5Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.length = f["raw"].shape[0]

        self.f     = None
        self.raw   = None
        self.label = None

    def _init_file(self):
        if self.f is None:
            self.f     = h5py.File(self.path, "r")
            self.raw   = self.f["raw"]
            self.label = self.f["label"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_file()
        x = torch.from_numpy(self.raw[idx]).float()
        y = torch.from_numpy(self.label[idx]).float()
        return x, y


# ─────────────────────────────
# LOSS
# ─────────────────────────────
def loss_fn(pred, target):
    return nn.functional.mse_loss(pred, target)


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":

    local_rank, device = setup_ddp()

    full = H5Dataset(H5_PATH)

    g        = torch.Generator().manual_seed(SEED)
    base_idx = torch.randperm(len(full), generator=g)

    train_idx = base_idx[:TRAIN_SIZE].tolist()
    val_idx   = base_idx[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE].tolist()

    if is_main():
        print(f"Dataset: {len(full)} total | {len(train_idx)} train | {len(val_idx)} val")

    train_ds = Subset(full, train_idx)
    val_ds   = Subset(full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=DistributedSampler(train_ds, shuffle=True),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=DistributedSampler(val_ds, shuffle=False),
        num_workers=4,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels=3,
        out_channels=1,
        f_maps=64,
        num_levels=3,
        final_sigmoid=False,
        is_segmentation=False,
    ).to(device)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler    = torch.amp.GradScaler()

    if is_main():
        writer   = SummaryWriter(os.path.join(RUN_DIR, EXP_NAME))
        log_file = open(os.path.join(LOG_DIR, f"{EXP_NAME}.log"), "a", buffering=1)

    ckpt_best = os.path.join(CKPT_DIR, f"best_unet_{EXP_NAME}.pth")
    ckpt_last = os.path.join(CKPT_DIR, f"last_net_{EXP_NAME}.pth")

    best_val = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        train_loader.sampler.set_epoch(epoch)
        model.train()

        train_loss = torch.tensor(0.0, device=device)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach()

        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss = (train_loss / dist.get_world_size() / len(train_loader)).item()

        # Validate
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda"):
                    val_loss += loss_fn(model(x), y).detach()

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = (val_loss / dist.get_world_size() / len(val_loader)).item()

        # Log
        if is_main():
            msg = f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}"
            print(msg)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            log_file.write(msg + "\n")

            torch.save(
                {"model": model.module.state_dict(), "epoch": epoch, "val": val_loss},
                ckpt_last,
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.module.state_dict(), ckpt_best)
                print(f"  → New best: {best_val:.6f}")

    if is_main():
        writer.close()
        log_file.close()
        print(f"\nDone. Best val loss: {best_val:.6f}")

    cleanup_ddp()