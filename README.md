# Cooldata - A Large-Scale Electronics Cooling 3D Flow Field Dataset
Cooldata is a large-scale electronics cooling dataset, containing over 60k stationary 3D flow fields for a diverse set of geometries, simulated with the commercial solver Simcenter STAR-CCM+. This library can be used to access the dataset and streamline its application in machine learning tasks.

![example case](docs/_static/case.png)

Find the documentation at [cooldata.readthedocs.io](https://cooldata.readthedocs.io/).

## Features
- **Data Storage:** Organized in folders containing `.cgns` files for compatibility with computational fluid dynamics tools.
- **PyVista Integration:** Access to dataset samples as PyVista objects for easy 3D visualization and manipulation.
- **Graph Neural Network Support:**
  - **DGL Support:**
    - Surface and volume data in mesh format.
    - 3D visualization of samples and predictions.
    - L2 loss computation and aggregate force evaluation for model training.
  - **PyG Support:** Implementing functionalities similar to DGL.
- **Hugging Face Integration:** Direct dataset loading from [Hugging Face](https://huggingface.co/).
- **Voxelized Flow Field Support:** Facilitates image processing-based ML approaches.
- **Comprehensive Metadata Accessibility:** All metadata is accessible through the library.
- **Metadata Filtering:** Explore and download samples by velocity, temperature, body count, position, and more — without downloading the full dataset.

## Installation

```bash
pip install cooldata
```

If you want to use the DGL support, you also need to install the [DGL](https://www.dgl.ai/) library, as documented [here](https://www.dgl.ai/pages/start.html).

## Dataset Overview

| | |
|---|---|
| Simulated samples | 60,848 |
| Runs | run_1, run_3, run_4, run_6, run_7 |
| Inlet velocity V | 1.0 – 7.0 m/s |
| Body temperatures T1–T6 | 20.0 – 80.0 °C |
| Bodies 1–2 | Quads — always active |
| Bodies 3–4 | Quads — sometimes active |
| Bodies 5–6 | Cylinders — sometimes active |

A body is **inactive** when its y-position equals the sentinel value `1.0`.

## Filtering & Downloading

The `MetadataFilter` class lets you explore the dataset and download only the samples you need, without touching any `.cgns` files until you're ready.

### Setup

Place these two files in your data directory:

```
Cooldataset/
    metadata.parquet       # design parameters for all 60,848 samples
```

### Explore before downloading

```python
from cooldata.metadata import MetadataFilter

f = MetadataFilter("Cooldataset/metadata.parquet")

# Print dataset summary: sample counts, velocity/temperature ranges, active bodies
f.summary()

# Check how many samples match a filter before committing to a download
print(f.velocity(min=4.0).n_bodies(min=3).count())
f.reset()

# Get matching metadata as a pandas DataFrame
df = (
    f.velocity(min=3.0, max=5.0)
     .temperature(body=1, min=60.0)
     .get_dataframe()
)
f.reset()
```

### Download by filter

```python
ds = (
    f.velocity(min=5.0)
     .temperature(body=1, min=70.0)
     .n_bodies(min=3)
     .load(num_samples=50)
)
f.reset()
```

### Download by specific design IDs

```python
ds = f.load_by_ids([125002, 125037, 212515])
```

### Download randomly

```python
ds = f.load_random(n=20, seed=42)

# Random from a filtered subset
ds = f.velocity(min=5.0).load_random(n=20, seed=42)
f.reset()
```

### Download by run

```python
# Available runs: run_1, run_3, run_4, run_6, run_7
ds = f.load_by_run("run_1", num_samples=100)
```

### Filter reference

| Method | Description | Example |
|---|---|---|
| `.velocity(min, max)` | Inlet velocity V | `.velocity(min=4.0, max=6.0)` |
| `.temperature(body, min, max)` | Body temperature. `body=None` = any body. | `.temperature(body=1, min=60.0)` |
| `.n_bodies(exactly, min, max)` | Total active bodies | `.n_bodies(exactly=4)` |
| `.n_quads(exactly, min, max)` | Active quads (bodies 1–4) | `.n_quads(min=2)` |
| `.n_cylinders(exactly, min, max)` | Active cylinders (bodies 5–6) | `.n_cylinders(exactly=1)` |
| `.position(body, x_min, x_max, y_min, y_max)` | Body position | `.position(body=1, x_max=0.5)` |
| `.size(quad, xs_min, xs_max, ys_min, ys_max)` | Quad dimensions | `.size(quad=1, xs_min=0.1)` |
| `.radius(cylinder, min, max)` | Cylinder radius (5 or 6) | `.radius(cylinder=5, max=0.1)` |
| `.run(*run_names)` | Restrict to specific runs | `.run("run_1", "run_3")` |
| `.custom(expr)` | Raw pandas query | `.custom("V > 4.5 and T1 < 40")` |
| `.reset()` | Clear all filters | `f.reset()` |
| `.count()` | Count matching samples | `f.velocity(min=4).count()` |
| `.get_design_ids()` | List matching design IDs | `f.n_bodies(exactly=2).get_design_ids()` |
| `.get_dataframe()` | Matching metadata as DataFrame | `f.velocity(min=4).get_dataframe()` |

> **Note:** Filters stack — always call `.reset()` between separate filter chains.

## Example Usage

See the `examples` folder for a detailed example of how to use the library.
