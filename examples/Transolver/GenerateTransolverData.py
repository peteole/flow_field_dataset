import gc

import h5py
import numpy as np

from cooldata.metadata import MetadataFilter

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
METADATA_PATH     = "../datasets/pyvista/metadata.parquet"
DATA_DIR          = "../datasets/pyvista"
OUT_PATH          = "cool_dataset_Transolver.h5"
NUM_SAMPLES       = 1000          # set to None to process all matching samples

# Optional filters — set to None to disable
FILTER_VELOCITY   = (1.0, 7.0)   # (min, max) or None
FILTER_N_BODIES   = None         # exact int or None

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

# Input dim: Normal(3) + body_feats(6×6=36) + inflow(1) = 40
INPUT_DIM = 3 + (MAX_BODIES * BODY_FEAT_DIM) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)


def normalize_coords(pts):
    """Normalise (N, 3) XYZ to [0, 1] using domain bounds."""
    out = pts.copy().astype(np.float32)
    out[:, 0] = normalize(out[:, 0], *DOMAIN_X)
    out[:, 1] = normalize(out[:, 1], *DOMAIN_Y)
    out[:, 2] = normalize(out[:, 2], *DOMAIN_Z)
    return out


def encode_metadata(metadata, N):
    """
    Encodes boundary conditions into a fixed-size per-point tensor.
    Returns (N, 37) float32 — broadcast to all surface points.
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

    body_flat  = np.array(body_rows, dtype=np.float32).ravel()
    inflow_vec = np.array(
        [normalize(metadata.inflow_velocity, VEL_MIN, VEL_MAX)],
        dtype=np.float32,
    )
    feat_vec = np.concatenate([body_flat, inflow_vec])   # (37,)
    return np.tile(feat_vec, (N, 1))                     # (N, 37)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_vlen(f, name, dtype=np.float32):
    if name not in f:
        f.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.vlen_dtype(dtype),
        )


def make_resizable(f, name, point_shape, dtype="int32"):
    if name not in f:
        f.create_dataset(
            name,
            shape=(0, *point_shape),
            maxshape=(None, *point_shape),
            dtype=dtype,
            chunks=(1, *point_shape),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load dataset via MetadataFilter
# ─────────────────────────────────────────────────────────────────────────────

print("Loading dataset ...")
f_filter = MetadataFilter(METADATA_PATH)

if FILTER_VELOCITY is not None:
    f_filter.velocity(min=FILTER_VELOCITY[0], max=FILTER_VELOCITY[1])
if FILTER_N_BODIES is not None:
    f_filter.n_bodies(exactly=FILTER_N_BODIES)

print(f"Filter matches {f_filter.count():,} samples.")

ds = f_filter.load(num_samples=NUM_SAMPLES, data_dir=DATA_DIR)
f_filter.reset()

print(f"Loaded {len(ds)} samples.")

# ─────────────────────────────────────────────────────────────────────────────
# Write HDF5
# ─────────────────────────────────────────────────────────────────────────────

with h5py.File(OUT_PATH, "a") as f:

    make_vlen(f, "coords")
    make_vlen(f, "input")
    make_vlen(f, "target")
    make_resizable(f, "n_points",   (1,), dtype="int32")
    make_resizable(f, "sample_ids", (1,), dtype="int32")

    coords_ds   = f["coords"]
    input_ds    = f["input"]
    target_ds   = f["target"]
    n_points_ds = f["n_points"]
    ids_ds      = f["sample_ids"]

    written = 0
    skipped = 0

    for i, sample in enumerate(ds.samples):
        print(f"Processing sample {i} (design {sample.design_id})", end="  ")

        try:
            # ── Extract surface data ──────────────────────────────────────
            combined  = sample.surface_data[0].combine(merge_points=True)
            positions = combined.cell_centers().points.astype(np.float32)          # (N, 3)
            normals   = combined.extract_surface().face_normals.astype(np.float32) # (N, 3)
            wall_temp = combined.cell_data["Temperature"].astype(np.float32)       # (N,)

            N_actual = positions.shape[0]
            print(f"N={N_actual}", end="  ")

            # ── Sanity check ──────────────────────────────────────────────
            if wall_temp.min() < 260 or wall_temp.max() > 380:
                print(f"SKIPPED (temp {wall_temp.min():.1f}–{wall_temp.max():.1f} K)")
                skipped += 1
                continue

            # ── Normalise coordinates ─────────────────────────────────────
            coords_norm = normalize_coords(positions)               # (N, 3)

            # ── Encode metadata ───────────────────────────────────────────
            meta_feats = encode_metadata(sample.metadata, N_actual) # (N, 37)

            # ── Assemble input: Normal(3) | body_feats(36) | inflow(1) ───
            input_feats = np.concatenate([
                normals,     # (N, 3)
                meta_feats,  # (N, 37)
            ], axis=1).astype(np.float32)                           # (N, 40)

            assert input_feats.shape == (N_actual, INPUT_DIM), \
                f"Shape mismatch: {input_feats.shape}"

            # ── Normalise target ──────────────────────────────────────────
            target = normalize(wall_temp, TEMP_MIN, TEMP_MAX) \
                         .reshape(-1, 1).astype(np.float32)         # (N, 1)

            # ── Write to HDF5 ─────────────────────────────────────────────
            slot = coords_ds.shape[0]
            coords_ds.resize(slot + 1,   axis=0)
            input_ds.resize(slot + 1,    axis=0)
            target_ds.resize(slot + 1,   axis=0)
            n_points_ds.resize(slot + 1, axis=0)
            ids_ds.resize(slot + 1,      axis=0)

            coords_ds[slot]   = coords_norm.ravel()   # (N*3,)
            input_ds[slot]    = input_feats.ravel()   # (N*40,)
            target_ds[slot]   = target.ravel()        # (N,)
            n_points_ds[slot] = [N_actual]
            ids_ds[slot]      = [sample.design_id]

            written += 1
            print(f"OK (written={written})")

            del combined, positions, normals, wall_temp
            del coords_norm, meta_feats, input_feats, target
            gc.collect()

        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1

    f.flush()

print(f"\nDone.")
print(f"  Written : {written}")
print(f"  Skipped : {skipped}")
print(f"  Total in file : {coords_ds.shape[0]}")