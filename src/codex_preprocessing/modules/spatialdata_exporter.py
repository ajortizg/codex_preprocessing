import abc
import logging
import os.path as osp
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import spatialdata as sd

from codex_preprocessing._constants import Keys
from codex_preprocessing.data import CodexDataset
from codex_preprocessing.utils import ensure_path

log = logging.getLogger(__name__)


class DataExporter(abc.ABC):
    def __init__(self):
        super().__init__()

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        return ds

    @abc.abstractmethod
    def __call__(self, ds: CodexDataset):
        pass


class SpatialDataExporter(DataExporter):
    """
    Export CODEX datasets to SpatialData format with optional Sopa Explorer output.

    This class converts processed CODEX multiplexed imaging data into the SpatialData
    format, which provides a standardized representation for spatial omics data. It
    supports marker filtering, renaming, multi-resolution pyramid generation, and
    optional export to Sopa Explorer format for interactive visualization.

    Args:
        out_dir: output directory for exported files. Will be created if it doesn't exist
        remove_markers: list of marker name substrings to exclude from export.
            Any marker containing these substrings will be filtered out
        save_zarr: if True, save the SpatialData object as a Zarr store. Defaults to True
        save_explorer: if True, save in Sopa Explorer format for visualization. Defaults to False
        scale_factors: list of downsampling factors for multi-resolution pyramids.
            For example, [2, 4, 8] creates additional resolution levels at 1/2, 1/4, and 1/8
            of the original resolution. Defaults to None (single resolution)
        rename_dict: dictionary mapping marker name substrings to replacement strings.
            For example, {"CD3-": "CD3"} would rename "CD3-FITC" to "CD3FITC". Defaults to None

    Note:
        - The exporter processes each region independently to handle large datasets
        - Output files are named using the pattern "reg{region:03d}.zarr" or "reg{region:03d}.explorer"
        - Existing output files will be overwritten automatically
        - CODEX metadata is preserved in the SpatialData object attributes
    """

    def __init__(
        self,
        out_dir: str | Path,
        remove_markers: Optional[List[str]],
        scale_factors: Optional[List[int]] = None,
        rename_dict: Optional[Dict[str, str]] = None,
    ):
        self.remove_markers = remove_markers or []
        self.out_dir = ensure_path(out_dir)
        self.scale_factors = scale_factors
        self.rename_dict = rename_dict or {}

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        ds.group_channels()
        ds.set_lazy(True)
        return ds

    def __call__(self, ds: CodexDataset):
        """
        Execute the export process for a CODEX dataset.

        This method serves as the main entry point for the export process. It
        prepares the dataset and processes each region independently to handle
        large datasets efficiently.

        Args:
            ds: the CODEX dataset to export. Must contain processed multiplexed
                imaging data with proper metadata and region information
        """

        pixel_size_um = ds.meta.resolution_nm / 1000

        regions = ds.get_unique_regions()
        log.info(f"Exporting {len(regions)} region(s): {regions}")

        for idx, region in enumerate(regions, 1):
            log.info(f"Processing region {region} ({idx}/{len(regions)})")

            reg_ds = ds.make_copy({"region": region})
            self._process_region(reg_ds, region, pixel_size_um)

        log.info("Export completed successfully")

    def _process_region(self, reg_ds: CodexDataset, region: int, pixel_size_um: float):
        """
        Process and export a single region from the CODEX dataset.

        This method handles the core conversion logic for transforming a single
        region of CODEX data into SpatialData format. It performs marker filtering,
        renaming, and creates the final SpatialData object with proper coordinate
        transformations and metadata.

        Args:
            reg_ds: region-specific dataset containing only data for the current region
            region: region identifier for naming output files
            pixel_size_um: pixel size in micrometers for spatial scaling

        Raises:
            ValueError: if no valid markers remain after filtering
        """
        img, markers = self._collect_images_and_markers(reg_ds)
        log.info(f"Collected {len(markers)} markers: {markers.tolist()}")

        # Filter markers
        keep_indices = self._get_valid_marker_indices(markers)
        img = img[keep_indices]
        markers = markers[keep_indices].tolist()

        # Rename markers
        if self.rename_dict is not None:
            for substr, newstr in self.rename_dict.items():
                markers = [newstr if substr in m else m for m in markers]

        markers = markers.tolist()
        if len(markers) == 0:
            raise ValueError(
                f"No valid markers remaining after filtering for region {region}. " f"Check 'remove_markers' parameter: {self.remove_markers}"
            )
        log.info(f"Final marker set ({len(markers)}): {markers}")

        sdata = self._create_spatialdata(img, markers, reg_ds)
        self._save_zarr(sdata, region)
        self._save_outputs(sdata, region, pixel_size_um)

    def _collect_images_and_markers(self, reg_ds: CodexDataset) -> Tuple[da.Array, np.ndarray]:
        img_list = []
        markers = []

        for data in reg_ds:
            img_list.append(data["img"])
            markers.extend([reg_ds.meta.marker_name(data["cycle"], ch) for ch in data["channel"]])
        return da.concatenate(img_list, axis=0), np.asarray(markers)

    def _get_valid_marker_indices(self, markers: np.ndarray) -> np.ndarray:
        mask = np.ones(markers.shape, dtype=bool)

        for rm in self.remove_markers:
            # np.char.find returns -1 if the substring is not found
            mask &= np.char.find(markers, rm) == -1

        return np.where(mask)[0]

    def _create_spatialdata(self, img: da.Array, markers: List[str], reg_ds: CodexDataset) -> sd.SpatialData:
        """
        Create a SpatialData object from the image and markers.

        Args:
            img: image array with shape (C, Y, X)
            markers: list of marker names
            reg_ds: region dataset for metadata

        Returns:
            SpatialData object with image and metadata
        """
        sdata = sd.SpatialData(
            images={
                Keys.IMAGE: sd.models.Image2DModel.parse(
                    img,
                    dims=("c", "y", "x"),
                    c_coords=markers,
                    transformations={Keys.DEFAULT_CS: sd.transformations.Identity()},
                    scale_factors=self.scale_factors,
                )
            }
        )

        sdata.attrs[Keys.CODEX_METADATA] = reg_ds.meta.get_dict()

        log.info(f"Created SpatialData object: {sdata}")
        return sdata

    def _save_zarr(self, sdata: sd.SpatialData, region: int):
        """Save as Zarr store."""
        zarr_path = self.out_dir / f"reg{region:03d}.zarr"

        # Remove existing directory
        if zarr_path.exists():
            log.info(f"Removing existing zarr directory: {zarr_path}")
            shutil.rmtree(zarr_path)

        sdata.write(str(zarr_path), overwrite=True)
        log.info(f"Zarr saved successfully: {zarr_path}")
