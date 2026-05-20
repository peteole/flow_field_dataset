import os
import gc
import numpy as np
import torch
import dgl
from cooldata.dgl_flow_field_dataset import DGLSurfaceFlowFieldDataset
from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH         = "../datasets/pyvista"
CACHE_DIR         = "./dgl_surface_cache"
NUM_TOTAL_SAMPLES = 1000

TEMP_MIN, TEMP_MAX = 293.15, 353.15   # K
VEL_MIN,  VEL_MAX  = 1.0,   7.0      # m/s

# Domain bounds
DOMAIN_X = (0.0, 0.5)
DOMAIN_Y = (0.0, 0.1)
DOMAIN_Z = (0.0, 0.02)

# Per-body metadata bounds
BODY_POS_X_MAX = 0.5
BODY_POS_Y_MAX = 0.1
BODY_POS_Z_MAX = 0.02
BODY_SIZE_MAX  = 0.05

# Heat source encoding
MAX_BODIES    = 6
BODY_FEAT_DIM = 6   # [x, y, z, temp, size_lateral, size_vertical]

# Node feature dimension:
#   Position(3) + Normal(3) + SurfaceType_onehot(5) + body_feats(36) + inflow(1) = 48
NODE_DIM = 3 + 3 + 5 + (MAX_BODIES * BODY_FEAT_DIM) + 1   # = 48

# Edge feature dimension:
#   dx(3) — normalised connectivity vector
EDGE_DIM = 3

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)

def normalize_coords(pts):
    """Normalise (N, 3) XYZ to [0, 1] using domain bounds."""
    out = pts.clone() if isinstance(pts, torch.Tensor) else torch.tensor(pts)
    out = out.float()
    out[:, 0] = (out[:, 0] - DOMAIN_X[0]) / (DOMAIN_X[1] - DOMAIN_X[0])
    out[:, 1] = (out[:, 1] - DOMAIN_Y[0]) / (DOMAIN_Y[1] - DOMAIN_Y[0])
    out[:, 2] = (out[:, 2] - DOMAIN_Z[0]) / (DOMAIN_Z[1] - DOMAIN_Z[0])
    return out

def normalize_dx(dx):
    """
    Normalise edge connectivity vectors by domain diagonal.
    Same fixed normalisation used for SDF in U-Net.
    """
    DOMAIN_DIAG = np.sqrt(0.5**2 + 0.1**2 + 0.02**2)   # ≈ 0.511 m
    return dx / DOMAIN_DIAG

# ─────────────────────────────────────────────────────────────────────────────
# Metadata encoder
# ─────────────────────────────────────────────────────────────────────────────
def encode_metadata(metadata, N):
    """
    Encodes boundary conditions into a fixed-size per-node tensor.
    Returns (N, 37) float32 tensor.
    """
    body_rows = []

    for q in metadata.quads:
        if q.position.y >= 1.0:
            continue
        body_rows.append([
            normalize(q.position.x,           0.0,      BODY_POS_X_MAX),
            normalize(q.position.y,           0.0,      BODY_POS_Y_MAX),
            normalize(q.position.z,           0.0,      BODY_POS_Z_MAX),
            normalize(q.temperature + 273.15, TEMP_MIN, TEMP_MAX),
            normalize(q.size_x,               0.0,      BODY_SIZE_MAX),
            normalize(q.size_z,               0.0,      BODY_SIZE_MAX),
        ])

    for c in metadata.cylinders:
        if c.position.y >= 1.0:
            continue
        body_rows.append([
            normalize(c.position.x,           0.0,      BODY_POS_X_MAX),
            normalize(c.position.y,           0.0,      BODY_POS_Y_MAX),
            normalize(c.position.z,           0.0,      BODY_POS_Z_MAX),
            normalize(c.temperature + 273.15, TEMP_MIN, TEMP_MAX),
            normalize(c.radius * 2,           0.0,      BODY_SIZE_MAX),
            normalize(c.height,               0.0,      BODY_SIZE_MAX),
        ])

    if len(body_rows) > MAX_BODIES:
        print(f"\n    Warning: {len(body_rows)} bodies, truncating to {MAX_BODIES}")
        body_rows = body_rows[:MAX_BODIES]

    while len(body_rows) < MAX_BODIES:
        body_rows.append([0.0] * BODY_FEAT_DIM)

    body_flat  = torch.tensor(body_rows, dtype=torch.float32).ravel()  # (36,)
    inflow_val = normalize(metadata.inflow_velocity, VEL_MIN, VEL_MAX)
    inflow_vec = torch.tensor([inflow_val], dtype=torch.float32)        # (1,)
    feat_vec   = torch.cat([body_flat, inflow_vec])                     # (37,)
    return feat_vec.unsqueeze(0).expand(N, -1).contiguous()             # (N, 37)

# ─────────────────────────────────────────────────────────────────────────────
# Node/edge feature builders
# ─────────────────────────────────────────────────────────────────────────────
def get_node_features(graph, metadata):
    """
    Build node feature matrix (N, 48).

    Layout:
        Position(3)     — normalised XYZ cell centres
        Normal(3)       — unit surface normal
        SurfaceType(5)  — one-hot: wall/inlet/outlet/symmetry/body
        body_feats(36)  — 6 heat sources × 6 features, zero-padded
        inflow(1)       — inlet velocity scalar
    """
    N = graph.num_nodes()

    pos_norm  = normalize_coords(graph.ndata["Position"])              # (N, 3)
    normals   = graph.ndata["Normal"].float()                          # (N, 3)
    surf_type = torch.nn.functional.one_hot(
        graph.ndata["SurfaceType"].long(), num_classes=5
    ).float()                                                          # (N, 5)
    meta_feats = encode_metadata(metadata, N)                         # (N, 37)

    node_X = torch.cat([
        pos_norm,    # (N, 3)
        normals,     # (N, 3)
        surf_type,   # (N, 5)
        meta_feats,  # (N, 37)
    ], dim=1)                                                          # (N, 48)

    assert node_X.shape == (N, NODE_DIM), \
        f"Node feature shape mismatch: {node_X.shape}"
    return node_X


def get_edge_features(graph):
    """
    Build edge feature matrix (E, 3).
    Normalise dx by domain diagonal for consistent scale.
    """
    dx = graph.edata["dx"].float()   # (E, 3)
    return normalize_dx(dx)          # (E, 3)


def get_node_target(graph):
    """Target: normalised wall Temperature only (N, 1)."""
    temp = graph.ndata["Temperature"].float()          # (N,)
    temp_norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    return temp_norm.unsqueeze(1)                      # (N, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Load datasets
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading PyVista dataset from {DATA_PATH} ...")
ds_pv = PyvistaFlowFieldDataset.load_from_huggingface(
    DATA_PATH, num_samples=NUM_TOTAL_SAMPLES
)

print(f"Converting to DGL surface graphs (cached at {CACHE_DIR}) ...")
os.makedirs(CACHE_DIR, exist_ok=True)
ds_dgl = DGLSurfaceFlowFieldDataset(CACHE_DIR, ds_pv, normalize=False)
print(f"Loaded {len(ds_dgl)} DGL surface graphs")

# ─────────────────────────────────────────────────────────────────────────────
# Build and save processed graphs
#
# Each graph is saved as a .pt file containing:
#   node_X  (N, 48)  — node features
#   edge_X  (E, 3)   — edge features
#   node_Y  (N, 1)   — target temperature
#   edges   (2, E)   — graph connectivity (src, dst)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = "./processed"  # Adjust this as needed
os.makedirs(OUT_DIR, exist_ok=True)

written = 0
skipped = 0

for i in range(NUM_TOTAL_SAMPLES):
    print(f"Processing sample {i}", end="  ")

    try:
        graph    = ds_dgl[i]
        metadata = ds_pv[i].metadata

        # ── Sanity check ──────────────────────────────────────────────────────
        temp = graph.ndata["Temperature"].float()
        if temp.min().item() < 260.0 or temp.max().item() > 360.0:
            print(f"SKIPPED (temp {temp.min():.1f}–{temp.max():.1f} K)")
            skipped += 1
            continue

        N = graph.num_nodes()
        E = graph.num_edges()
        print(f"N={N}  E={E}", end="  ")

        # ── Build features ────────────────────────────────────────────────────
        node_X = get_node_features(graph, metadata)   # (N, 48)
        edge_X = get_edge_features(graph)             # (E, 3)
        node_Y = get_node_target(graph)               # (N, 1)

        src, dst = graph.edges()
        edges = torch.stack([src, dst], dim=0)        # (2, E)

        # ── Save as .pt file ──────────────────────────────────────────────────
        out_file = os.path.join(OUT_DIR, f"sample_{i:05d}.pt")
        torch.save({
            "node_X":    node_X,
            "edge_X":    edge_X,
            "node_Y":    node_Y,
            "edges":     edges,
            "sample_id": i,
        }, out_file)

        written += 1
        print(f"OK (written={written})")

        del graph, node_X, edge_X, node_Y, edges
        gc.collect()

    except Exception as e:
        print(f"ERROR: {e}")
        skipped += 1

print(f"\nDone.")
print(f"  Written : {written}")
print(f"  Skipped : {skipped}")
print(f"  Output  : {OUT_DIR}")