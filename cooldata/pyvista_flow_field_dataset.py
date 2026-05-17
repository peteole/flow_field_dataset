import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import huggingface_hub as hf  # type: ignore
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm

from cooldata.metadata import SystemParameters, df_row_to_system_parameters

VolumeFieldType = Literal[
    "Velocity_0",
    "Velocity_1",
    "Velocity_2",
    "Pressure",
    "Temperature",
    "TurbulentKineticEnergy",
    "TurbulentDissipationRate",
]


SurfaceFieldType = Literal[
    "AreaMagnitude",
    "HeatTransferCoefficient",
    "Normal_0",
    "Normal_1",
    "Normal_2",
    "Temperature",
    "Pressure",
    "WallShearStressMagnitude",
    "WallShearStress_0",
    "WallShearStress_1",
    "WallShearStress_2",
]


class PyvistaSample:
    """
    A PyVista sample representing a 3D flow field dataset. The data is stored on disk and can be loaded to a PyVista object when needed
    """
    def __init__(
        self,
        volume_path: str | Path,
        surface_path: str | Path,
        metadata: SystemParameters | None = None,
    ):
        self.volume_path = Path(volume_path)
        self.surface_path = Path(surface_path)
        self._volume_data: pv.MultiBlock | None = None
        self._surface_data: pv.MultiBlock | None = None
        self.metadata = metadata
        self._bounds: tuple[float, float, float, float, float, float] | None = None

    @property
    def surface_data(self):
        if self._surface_data is None:
            self._surface_data = pv.read(self.surface_path)
        # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._surface_data

    @property
    def volume_data(self):
        if self._volume_data is None:
            self._volume_data = pv.read(self.volume_path)
        # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._volume_data

    def plot_surface(self, field: SurfaceFieldType):
        self.surface_data.plot(scalars=field)

    def plot_volume(self, field: VolumeFieldType):
        self.volume_data[0][0][0].plot(scalars=field, opacity=0.7)

    def get_points(self) -> np.ndarray:
        """
        Returns the points of the dataset as a numpy array.

        Returns:
        --------
        np.ndarray: The points of the dataset. Shape: (n_points, 3)
        """
        # TODO check if same works for non cgns
        block = self.volume_data[0][0][0]
        return block.cell_centers().points

    def get_surface_points(self, block_index: int) -> np.ndarray:
        """
        Returns the points of the surface dataset as a numpy array.

        Returns:
        --------
        np.ndarray: The points of the surface dataset. Shape: (n_points, 3)
        """
        block = self.surface_data[0][block_index]
        return block.points

    def get_labeled_surface_points(self) -> np.ndarray:
        """
        Returns the surface points of the dataset with their block index as a numpy array.

        Returns:
        --------
        np.ndarray: The labeled points of the dataset. Shape: (n_points, 4)
        """
        labeled_points = []
        for i, block in enumerate(self.surface_data[0]):
            labeled_points.append(
                np.hstack((block.points, np.full((block.n_points, 1), i)))
            )
        return np.vstack(labeled_points)

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the volume and surface data are loaded, False otherwise.
        """
        return self._volume_data is not None and self._surface_data is not None

    def load(self):
        self.volume_data
        self.surface_data

    def unload(self):
        # TODO: Test if this frees the memory
        self._volume_data = None
        self._surface_data = None

    def get_bounding_box(self):
        """
        Returns the bounding box of the volume data.
        The bounding box is a six-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        if self._bounds is not None:
            return self._bounds
        was_loaded = self.is_loaded
        self._bounds = self.volume_data.bounds
        if not was_loaded:
            self.unload()
        return self._bounds

    @property
    def design_id(self) -> int:
        """
        Returns the design ID of the sample, which is extracted from the file name.
        The design ID is assumed to be the second last part of the file name, split by underscores.
        """
        stem = self.volume_path.stem
        id = re.search(r"_(\d+)", stem)
        if id:
            return int(id.group(1))
        raise ValueError("Design ID not found")

    def _repr_html_(self) -> str:
        if not hasattr(self, "metadata") or self.metadata is None:
            return "<p>No metadata available for this sample to generate HTML representation.</p>"

        # Ensure metadata object is not None and has quads and cylinders attributes
        if not hasattr(self.metadata, "quads") or not hasattr(
            self.metadata, "cylinders"
        ):
            return "<p>Metadata object is malformed (missing quads or cylinders attributes).</p>"

        _design_id_str: str
        try:
            _design_id_str = str(self.design_id)
        except ValueError:
            _design_id_str = "N_A"  # Use a safe fallback for ID

        try:
            bounds = self.get_bounding_box()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        except Exception as e:
            return f"""<p>Error getting bounding box for sample ID {_design_id_str}: {e}. 
Make sure volume data is accessible and valid.</p>"""

        xmin, xmax, ymin, ymax, _, _ = bounds

        world_width = xmax - xmin
        world_height = ymax - ymin

        if (
            world_width <= 1e-6 or world_height <= 1e-6
        ):  # Using a small epsilon for float comparison
            return f"""<p>Bounding box for sample ID {_design_id_str} has zero or negligible 2D dimensions 
(width: {world_width:.2e}, height: {world_height:.2e}). Cannot render.</p>"""

        # SVG viewport dimensions (fixed size for the output image)
        svg_width = 300
        svg_height = 200

        padding = 0.05 * max(world_width, world_height)

        view_box_xmin = xmin - padding
        view_box_ymin = ymin - padding
        view_box_width = world_width + 2 * padding
        view_box_height = world_height + 2 * padding

        # Determine a reasonable stroke width based on world dimensions
        # This aims for a stroke that's roughly 0.5% of the main dimension
        dynamic_stroke_width = 0.005 * max(world_width, world_height)

        html_parts = [
            '<div style="font-family: Arial, sans-serif; margin: 10px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #fdfdfd;">',
            f'  <h4 style="margin-top:0; margin-bottom: 10px; color: #333;">Design ID: {_design_id_str}</h4>',
            f'  <svg width="{svg_width}" height="{svg_height}" viewBox="{view_box_xmin} {view_box_ymin} {view_box_width} {view_box_height}" style="border:1px solid #ccc; background-color: #ffffff;">',
        ]

        # Unique ID for clipPath, incorporating object id for robustness in some edge cases
        clip_path_id = f"clipPath_PyvistaSample_{_design_id_str}_{hex(id(self))[-6:]}"

        html_parts.append("    <defs>")
        html_parts.append(f'      <clipPath id="{clip_path_id}">')
        html_parts.append(
            f'        <rect x="{xmin}" y="{ymin}" width="{world_width}" height="{world_height}" />'
        )
        html_parts.append("      </clipPath>")
        html_parts.append("    </defs>")

        html_parts.append(
            f'    <rect x="{xmin}" y="{ymin}" width="{world_width}" height="{world_height}" fill="none" stroke="#bbbbbb" stroke-width="{dynamic_stroke_width}" stroke-dasharray="{2 * dynamic_stroke_width},{2 * dynamic_stroke_width}" />'
        )

        html_parts.append(f'    <g clip-path="url(#{clip_path_id})">')

        quad_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, quader in enumerate(self.metadata.quads):
            q_x = quader.position.x - quader.size_x / 2
            q_y = quader.position.y - quader.size_y / 2
            color = quad_colors[i % len(quad_colors)]
            html_parts.append(
                f'      <rect x="{q_x}" y="{q_y}" width="{quader.size_x}" height="{quader.size_y}" fill="{color}" fill-opacity="0.6" stroke="{color}" stroke-opacity="0.9" stroke-width="{0.5 * dynamic_stroke_width}">'
            )
            html_parts.append(
                f"        <title>Quader {i + 1}\\nTemp: {quader.temperature:.1f}\\nPos: ({quader.position.x:.2f}, {quader.position.y:.2f})\\nSize: ({quader.size_x:.2f}, {quader.size_y:.2f})</title>"
            )
            html_parts.append("      </rect>")

        cyl_colors = ["#17becf", "#bcbd22", "#e377c2", "#7f7f7f", "#aec7e8", "#ffbb78"]
        for i, cylinder in enumerate(self.metadata.cylinders):
            color = cyl_colors[i % len(cyl_colors)]
            html_parts.append(
                f'      <circle cx="{cylinder.position.x}" cy="{cylinder.position.y}" r="{cylinder.radius}" fill="{color}" fill-opacity="0.6" stroke="{color}" stroke-opacity="0.9" stroke-width="{0.5 * dynamic_stroke_width}">'
            )
            html_parts.append(
                f"        <title>Cylinder {i + 1}\\nTemp: {cylinder.temperature:.1f}\\nPos: ({cylinder.position.x:.2f}, {cylinder.position.y:.2f})\\nRadius: {cylinder.radius:.2f}</title>"
            )
            html_parts.append("      </circle>")

        html_parts.append("    </g>")
        html_parts.append("  </svg>")

        html_parts.append(
            f'  <p style="font-size: 0.8em; color: #666; margin-top: 8px; margin-bottom: 0;">Bounding Box (xmin, xmax, ymin, ymax): ({xmin:.2f}, {xmax:.2f}, {ymin:.2f}, {ymax:.2f})</p>'
        )
        html_parts.append("</div>")

        return "\n".join(html_parts)


class PyvistaFlowFieldDataset:
    """
    The main class for working with the cooldata dataset.
    """
    def __init__(self, samples: list[PyvistaSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def slice(self, start: int, end: int):
        """
        Returns a slice of the dataset.
        """
        return PyvistaFlowFieldDataset(self.samples[start:end])

    def shuffle(self):
        """
        Shuffles the dataset in place.
        """
        np.random.shuffle(self.samples)

    @classmethod
    def try_from_directory(
        cls, data_dir: str | Path, num_samples: int
    ) -> "None | PyvistaFlowFieldDataset":
        data_dir = os.path.abspath(data_dir)
        data_dir = Path(data_dir)
        volume_dir = data_dir / "volume"
        surface_dir = data_dir / "surface"
        metadata_file = data_dir / "metadata.parquet"
        if not metadata_file.exists():
            print(f"Metadata file not found at {metadata_file}.")
            return None
        os.makedirs(volume_dir, exist_ok=True)
        os.makedirs(surface_dir, exist_ok=True)
        volume_files = list(volume_dir.glob("*.cgns"))
        surface_files = list(surface_dir.glob("*.cgns"))
        volume_indices = [int(f.stem.split("_")[-2]) for f in volume_files]
        surface_indices = [int(f.stem.split("_")[-2]) for f in surface_files]
        volume_indices.sort()
        surface_indices.sort()
        if volume_indices != surface_indices:
            return None
        volume_files = sorted(volume_files, key=lambda x: int(x.stem.split("_")[-2]))
        surface_files = sorted(surface_files, key=lambda x: int(x.stem.split("_")[-2]))
        samples = [PyvistaSample(v, s) for v, s in zip(volume_files, surface_files)]
        if len(samples) < num_samples:
            return None
        samples = samples[:num_samples]
        ds = cls(samples)
        metadata_df = pd.read_parquet(metadata_file)
        ds.add_metadata(metadata_df)
        print(f"Loaded {len(ds)} samples from '{data_dir}'.")
        return ds

    @classmethod
    def load_from_huggingface(
        cls, data_dir: str | Path, num_samples=3
    ) -> "PyvistaFlowFieldDataset":
        """
        Download the given number of samples from huggingface to data_dir.
        Args:
        - data_dir: The directory to download the data to.
        - num_samples: The number of samples to download
        """
        import warnings
        warnings.warn(
            "load_from_huggingface() is deprecated. "
            "Use MetadataFilter from cooldata.metadata instead:\n"
            "  from cooldata.metadata import MetadataFilter\n"
            "  ds = MetadataFilter('path/to/metadata.parquet').load_random(n=num_samples)",
            FutureWarning,
            stacklevel=2,
        )
        loaded = cls.try_from_directory(data_dir, num_samples)
        if loaded is not None:
            print(f"Loaded {len(loaded)} samples from '{data_dir}'.")
            return loaded

        data_dir = os.path.abspath(data_dir)
        data_dir = Path(data_dir)
        volume_dir = data_dir / "volume"
        surface_dir = data_dir / "surface"
        tmp_dir = data_dir / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        # remove existing files
        if os.path.exists(volume_dir):
            shutil.rmtree(volume_dir)
        if os.path.exists(surface_dir):
            shutil.rmtree(surface_dir)
        os.makedirs(volume_dir, exist_ok=True)
        os.makedirs(surface_dir, exist_ok=True)

        repo_id = "datasets/bgce/cooldata-v2"
        fs = hf.HfFileSystem()
        # download metadata file
        metadata_file = f"{repo_id}/metadata.parquet"
        local_metadata_path = os.path.join(data_dir, "metadata.parquet")
        fs.download(metadata_file, local_metadata_path)
        metadata_df = pd.read_parquet(local_metadata_path)

        runs = fs.glob(f"{repo_id}/runs/run_*", detail=False)
        samples: list[PyvistaSample] = []
        runs = sorted(runs, key=lambda x: int(x.split("/")[-1].removeprefix("run_")))
        for run in runs:
            run_name = os.path.basename(run)
            zip_files_in_run = list(fs.glob(f"{run}/batch_*.zip", detail=False))
            zip_files_in_run = sorted(
                zip_files_in_run,
                key=lambda x: int(
                    x.split("/")[-1].removeprefix("batch_").removesuffix(".zip")
                ),
            )
            for zip_file in zip_files_in_run:
                local_path = os.path.join(tmp_dir, run_name, os.path.basename(zip_file))
                try:
                    fs.download(zip_file, local_path)
                except Exception as e:
                    print(f"Failed to download {zip_file} for run {run_name}: {e}")
                    continue
                # Extract the zip file
                unzip_dir = os.path.join(
                    tmp_dir, run_name, os.path.basename(zip_file).removesuffix(".zip")
                )
                os.makedirs(unzip_dir, exist_ok=True)
                shutil.unpack_archive(local_path, unzip_dir)
                # Match indices
                volume_files = list(Path(unzip_dir).glob("volume_design_*_p.cgns"))
                surface_files = list(Path(unzip_dir).glob("surface_design_*_p.cgns"))
                volume_indices = [int(f.stem.split("_")[-2]) for f in volume_files]
                surface_indices = [int(f.stem.split("_")[-2]) for f in surface_files]
                volume_indices.sort()
                surface_indices.sort()
                if volume_indices != surface_indices:
                    print(f"Skipping {run_name} due to mismatched indices.")
                    continue
                volume_files = sorted(
                    volume_files, key=lambda x: int(x.stem.split("_")[-2])
                )
                surface_files = sorted(
                    surface_files, key=lambda x: int(x.stem.split("_")[-2])
                )
                for v, s in zip(volume_files, surface_files):
                    # Copy the files to the new directory
                    shutil.copy(v, volume_dir)
                    shutil.copy(s, surface_dir)
                    moved_volume_path = volume_dir / v.name
                    moved_surface_path = surface_dir / s.name
                    samples.append(PyvistaSample(moved_volume_path, moved_surface_path))
                    if len(samples) >= num_samples:
                        shutil.rmtree(tmp_dir)
                        ds = cls(samples)
                        ds.add_metadata(metadata_df)
                        print(f"Loaded {len(ds)} samples from '{data_dir}'.")
                        return ds
        # Clean up temporary directory
        shutil.rmtree(tmp_dir)

        ds = cls(samples)
        ds.add_metadata(metadata_df)
        print(f"Loaded {len(ds)} samples from '{data_dir}'.")
        return ds

    def add_metadata(self, metadata_df: pd.DataFrame) -> None:
        """
        Adds metadata to the samples from a pandas DataFrame.
        The DataFrame should have a column 'design_id' that matches the design ID of the samples.
        """
        if not isinstance(metadata_df, pd.DataFrame):
            raise TypeError("metadata_df must be a pandas DataFrame.")

        for sample in self.samples:
            design_id = sample.design_id
            try:
                md = df_row_to_system_parameters(metadata_df, design_id)
                sample.metadata = md
            except Exception as e:
                print(
                    f"Failed to add metadata for sample with design ID {design_id}: {e}"
                )

    def get_bounds(self):
        """
        Returns the bounding box of the volume data.
        The bounding box is a six-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        bounds = (np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf)
        self.unload()

        with ProcessPoolExecutor() as executor:
            sample_bounds_list = list(
                tqdm(
                    executor.map(get_sample_bounds, self.samples),
                    total=len(self.samples),
                    desc="Computing bounds",
                )
            )

        for sample_bounds in sample_bounds_list:
            bounds = (
                min(bounds[0], sample_bounds[0]),
                max(bounds[1], sample_bounds[1]),
                min(bounds[2], sample_bounds[2]),
                max(bounds[3], sample_bounds[3]),
                min(bounds[4], sample_bounds[4]),
                max(bounds[5], sample_bounds[5]),
            )

        return bounds

    def load_to_memory(self):
        """
        Load all samples into memory.
        """
        for sample in self.samples:
            sample.load()

    def unload(self):
        """
        Unload all samples from memory.
        """
        for sample in self.samples:
            sample.unload()


def get_sample_bounds(sample):
    return sample.get_bounding_box()
