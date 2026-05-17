import gc

import h5py
import numpy as np
from scipy.spatial import cKDTree

from cooldata.metadata import MetadataFilter

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
METADATA_PATH = "datasets/pyvista/metadata.parquet"
DATA_DIR      = "datasets/pyvista"
OUT_PATH      = "cool_dataset.h5"
NUM_SAMPLES   = 1000

RESOLUTION     = 64
TEMP_MIN, TEMP_MAX = 293.15, 353.15
VEL_MIN,  VEL_MAX  = 1.0,   7.0

# Optional filters — set to None to disable
FILTER_VELOCITY = None   # e.g. (4.0, 7.0)
FILTER_N_BODIES = None   # e.g. 4


# ─────────────────────────────────────────────────────────────────────────────
# Grid (created once)
# ─────────────────────────────────────────────────────────────────────────────
def create_grid():
    x = np.linspace(0.0, 0.5,  RESOLUTION)
    y = np.linspace(0.0, 0.1,  RESOLUTION)
    z = np.linspace(0.0, 0.02, RESOLUTION)
    return np.meshgrid(x, y, z, indexing="ij")

xv, yv, zv = create_grid()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)


def compute_sdf(metadata, resolution, xv, yv, zv):
    """Compute signed distance field — negative inside bodies, positive outside."""
    sdf = np.full((resolution,) * 3, fill_value=np.inf, dtype=np.float32)

    for q in metadata.quads:
        if q.position.y >= 1.0:
            continue
        cx, cy, cz = q.position.x, q.position.y, q.position.z
        sx, sy, sz = q.size_x, q.size_y, q.size_z

        dx = np.maximum(np.abs(xv - cx) - sx / 2, 0)
        dy = np.maximum(np.abs(yv - cy) - sy / 2, 0)
        dz = np.maximum(np.abs(zv - cz) - sz / 2, 0)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        inside = (
            (np.abs(xv - cx) <= sx / 2) &
            (np.abs(yv - cy) <= sy / 2) &
            (np.abs(zv - cz) <= sz / 2)
        )
        dist[inside] *= -1
        sdf = np.minimum(sdf, dist)

    for c in metadata.cylinders:
        if c.position.y >= 1.0:
            continue
        cx, cy, cz = c.position.x, c.position.y, c.position.z

        r_xy = np.sqrt((xv - cx)**2 + (yv - cy)**2)
        d_r  = r_xy - c.radius
        d_z  = np.abs(zv - cz) - c.height / 2

        outside_r = d_r > 0
        outside_z = d_z > 0
        dist = np.where(
            outside_r & outside_z, np.sqrt(d_r**2 + d_z**2),
            np.where(outside_r, d_r,
            np.where(outside_z, d_z,
            np.maximum(d_r, d_z)))
        )
        sdf = np.minimum(sdf, dist)

    DOMAIN_DIAG = np.sqrt(0.5**2 + 0.1**2 + 0.02**2)
    sdf = np.clip(sdf / DOMAIN_DIAG, -1.0, 1.0)
    return sdf


def compute_boundary_channels(metadata, xv, yv, zv):
    body_temp = np.zeros((RESOLUTION,) * 3, dtype=np.float32)
    inflow    = np.zeros((RESOLUTION,) * 3, dtype=np.float32)

    for q in metadata.quads:
        if q.position.y >= 1.0:
            continue
        cx, cy, cz = q.position.x, q.position.y, q.position.z
        sx, sy, sz = q.size_x, q.size_y, q.size_z
        inside = (
            (np.abs(xv - cx) <= sx / 2) &
            (np.abs(yv - cy) <= sy / 2) &
            (np.abs(zv - cz) <= sz / 2)
        )
        body_temp[inside] = normalize(q.temperature + 273.15, TEMP_MIN, TEMP_MAX)

    for c in metadata.cylinders:
        if c.position.y >= 1.0:
            continue
        cx, cy, cz = c.position.x, c.position.y, c.position.z
        r_xy   = np.sqrt((xv - cx)**2 + (yv - cy)**2)
        inside = (r_xy <= c.radius) & (np.abs(zv - cz) <= c.height / 2)
        body_temp[inside] = normalize(c.temperature + 273.15, TEMP_MIN, TEMP_MAX)

    inflow[0, :, :] = normalize(metadata.inflow_velocity, VEL_MIN, VEL_MAX)
    return body_temp, inflow


def voxelize(points, values, xv, yv, zv):
    grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)
    tree = cKDTree(points)
    _, idx = tree.query(grid_points, workers=-1)
    return values[idx].reshape((RESOLUTION,) * 3)


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
    if "raw" not in f:
        f.create_dataset(
            "raw",
            shape=(0, 3, RESOLUTION, RESOLUTION, RESOLUTION),
            maxshape=(None, 3, RESOLUTION, RESOLUTION, RESOLUTION),
            dtype="float32",
            chunks=(1, 3, RESOLUTION, RESOLUTION, RESOLUTION),
        )
        f.create_dataset(
            "label",
            shape=(0, 1, RESOLUTION, RESOLUTION, RESOLUTION),
            maxshape=(None, 1, RESOLUTION, RESOLUTION, RESOLUTION),
            dtype="float32",
            chunks=(1, 1, RESOLUTION, RESOLUTION, RESOLUTION),
        )

    raw_ds   = f["raw"]
    label_ds = f["label"]

    written = 0
    skipped = 0

    for i, sample in enumerate(ds.samples):
        print(f"Processing sample {i} (design {sample.design_id})", end="  ")

        try:
            polydata = sample.volume_data[0][0][0].cell_data_to_point_data()
            temps    = np.array(polydata.point_data["Temperature"])
            points   = np.array(polydata.points)

            body_temp, inflow = compute_boundary_channels(sample.metadata, xv, yv, zv)
            sdf               = compute_sdf(sample.metadata, RESOLUTION, xv, yv, zv)

            raw     = np.stack([sdf, body_temp, inflow], axis=0).astype(np.float32)
            T_field = voxelize(points, temps, xv, yv, zv)

            if T_field.min() < 260 or T_field.max() > 380:
                print(f"SKIPPED (temp {T_field.min():.1f}–{T_field.max():.1f} K)")
                skipped += 1
                continue

            T_field = normalize(T_field, TEMP_MIN, TEMP_MAX)
            label   = T_field[np.newaxis, ...].astype(np.float32)

            idx = raw_ds.shape[0]
            raw_ds.resize(idx + 1,   axis=0)
            label_ds.resize(idx + 1, axis=0)
            raw_ds[idx]   = raw
            label_ds[idx] = label

            written += 1
            print(f"OK (written={written})")

            del polydata, points, T_field, raw, label, sdf, body_temp, inflow
            gc.collect()

        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1

    f.flush()

print(f"\nDone.")
print(f"  Written : {written}")
print(f"  Skipped : {skipped}")
print(f"  Total in file : {raw_ds.shape[0]}")