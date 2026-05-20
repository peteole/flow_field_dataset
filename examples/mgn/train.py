import os
import logging
import contextlib
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.models.meshgraphnet import MeshGraphNet
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_DIR = "./processed"
RUN_SIZES = [50000]

NODE_DIM = 40
EDGE_DIM = 3
OUT_DIM  = 1

HIDDEN_DIM          = 128
NUM_LAYERS_ENCODER  = 2
NUM_LAYERS_DECODER  = 2
NUM_MESSAGE_PASSING = 15
AGGREGATION         = "sum"

ACCUMULATION_STEPS = 1
LR     = 1e-4
EPOCHS = 200
TEMP_RANGE = 353.15 - 293.15

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
def setup_logger(run_size):
    os.makedirs("logs", exist_ok=True)

    log_file = f"logs/train_mgn_{run_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(f"mgn_{run_size}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ─────────────────────────────────────────────────────────────
# DDP
# ─────────────────────────────────────────────────────────────
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device("cuda", local_rank)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class MGNDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("sample_*.pt"))
        assert len(self.files) > 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")


# ─────────────────────────────────────────────────────────────
# Forward helper
# ─────────────────────────────────────────────────────────────
def forward_mgn(model, batch, device):
    node_X = batch["node_X"].to(device)
    edge_X = batch["edge_X"].to(device)
    node_Y = batch["node_Y"].to(device)
    edge_index = batch["edges"].to(device)

    graph = Data(x=node_X, edge_index=edge_index, edge_attr=edge_X)
    pred = model(node_X, edge_X, graph)

    return pred, node_Y


# ─────────────────────────────────────────────────────────────
# Training for ONE dataset size
# ─────────────────────────────────────────────────────────────
def run_experiment(full_dataset, indices, number_samples, device, local_rank):

    logger = setup_logger(number_samples) if is_main() else None
    writer = SummaryWriter(f"runs/mgn_{number_samples}") if is_main() else None

    if is_main():
        logger.info(f"\n===== RUN SIZE {number_samples} =====")

    # ── Subset
    subset = torch.utils.data.Subset(
        full_dataset,
        indices[:number_samples]
    )

    n_total = len(subset)
    n_train = int(0.90 * n_total)
    n_val   = n_total - n_train

    train_ds, val_ds = torch.utils.data.random_split(
        subset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(32)
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=48, sampler=train_sampler,
        num_workers=4, persistent_workers=True,
        collate_fn=lambda x: x[0]
    )

    val_loader = DataLoader(
        val_ds, batch_size=48, sampler=val_sampler,
        num_workers=4, persistent_workers=True,
        collate_fn=lambda x: x[0]
    )

    if is_main():
        logger.info(f"Train: {n_train} | Val: {n_val}")

    # ── Model
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
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")

    ckpt_name = f"checkpoint_{number_samples}.pth"
    best_name = f"mgn_best_{number_samples}.pth"

    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):

            with torch.amp.autocast("cuda"):
                pred, node_Y = forward_mgn(model, batch, device)
                loss = criterion(pred, node_Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()

        # ── Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda"):
                    pred, node_Y = forward_mgn(model, batch, device)
                    loss = criterion(pred, node_Y)

                val_loss += loss.item()

        # ── Reduce
        world_size = dist.get_world_size()

        t = torch.tensor([train_loss, val_loss], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

        train_loss = t[0].item() / (len(train_loader) * world_size)
        val_loss   = t[1].item() / (len(val_loader) * world_size)

        scheduler.step(val_loss)

        if is_main():
            logger.info(
                f"Epoch {epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f} "
            )

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            torch.save({
                "epoch": epoch,
                "model": model.module.state_dict(),
            }, ckpt_name)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.module.state_dict(), best_name)

    if is_main():
        writer.close()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():

    local_rank, device = setup_ddp()

    torch.manual_seed(32)

    full_dataset = MGNDataset(DATA_DIR)

    # fixed indices (important!)
    indices = torch.randperm(len(full_dataset))

    for size in RUN_SIZES:
        run_experiment(full_dataset, indices, size, device, local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()