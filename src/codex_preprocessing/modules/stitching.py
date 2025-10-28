import abc
import logging
import os
import os.path as osp
import re
import shutil
import subprocess
from copy import deepcopy
from glob import glob
from itertools import chain, repeat
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from codex_preprocessing._constants import CodexFiles
from codex_preprocessing.data import CodexDataset
from codex_preprocessing.modules.m2stitch import stitch_images
from codex_preprocessing.utils import ensure_path

log = logging.getLogger(__name__)


class Stitching(abc.ABC):
    """
    Abstract base class for CODEX image stitching algorithms.
    """

    def __init__(self, out_dir: Path | str, is_cmd: bool):
        super().__init__()
        self.is_cmd = is_cmd
        self.set_out_dir(out_dir) if out_dir is not None else None

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass

    def set_out_dir(self, out_dir: Path | str):
        """
        Set the output directory for the stitching process.

        Args:
            out_dir (Path | str): Output directory for stitched images.
        """
        self.out_dir = ensure_path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)


class Ashlar(Stitching):
    """
    CODEX image stitching using Ashlar software.

    This class provides a Python interface to the Ashlar image stitching software
    for combining CODEX image tiles into complete, seamless images. It handles
    command generation, execution, and output formatting for integration with
    the CODEX processing pipeline.

    Args:
        ds (CodexDataset): Input dataset containing raw CODEX images
        out_dir (Path | str): Output directory for stitched images
        align_channel (int): Channel number (1-indexed) to use for alignment
        max_shift (int): Maximum allowed shift in pixels for alignment
        filter_sigma (float): Gaussian filter sigma for preprocessing
        ome (bool): Whether to output OME-TIFF format
        layout (str, optional): Tile layout specification. If None, uses metadata
    """

    def __init__(
        self,
        ds: CodexDataset,
        out_dir: Path | str,
        align_channel: int,
        max_shift: int,
        filter_sigma: float,
        ome: bool,
        layout: Optional[str] = None,
    ):
        super().__init__(out_dir, is_cmd=True)

        self.layout = layout
        self.ome = ome
        self.align_channel = align_channel - 1
        self.max_shift = max_shift
        self.filter_sigma = filter_sigma
        self.ds = deepcopy(ds)
        self.meta = self.ds.meta
        self.extra_args = None

        paths = pd.Series(sum(self.ds.paths, []))  # flatten the list
        matches = paths.str.extract(CodexFiles.RAW_DIR_RE)
        matches = matches.astype(int)
        dir_paths = paths.map(osp.dirname)

        self.df = (
            pd.DataFrame({"cycle": matches["cycle"], "region": matches["region"], "path": dir_paths})
            .drop_duplicates(subset=["cycle", "region"])
            .sort_values(["cycle", "region"])
            .reset_index(drop=True)
        )

        self.FILESERIES_TEMPLATE = '"fileseries|{path}|pattern={region}_{series}_Z001_CH{channel}.tif|width={width}|height={height}|overlap={overlap}|layout={layout}|pixel_size={pixel_size}"'

    def set_extra_args(self, extra_args: str):
        """
        Set additional command-line arguments for the Ashlar stitching process.

        Args:
            extra_args (str): Additional arguments to include in the Ashlar command.
        """
        self.extra_args = extra_args

    def __call__(self, **kwargs):
        """
        Execute Ashlar stitching for all regions in the dataset.

        Runs the Ashlar stitching software for each region independently,
        creating seamless images from overlapping tiles. The method generates
        appropriate Ashlar commands and executes them via subprocess.

        Args:
            extra_args (str, optional): Additional command-line arguments to pass
                to Ashlar. These are appended to the generated command.
        """
        for region in self.df["region"].unique():
            cmd = self.create_cmd(region)

            if self.extra_args:
                cmd += " " + self.extra_args

            log.info(cmd)
            log.info(f"Runing ashlar for region: {region}")
            try:
                subprocess.run(cmd, check=True, shell=True)
                log.info("ashlar command executed successfully.")
            except subprocess.CalledProcessError as e:
                log.error(f"Error occurred while executing ashlar: {e}")

    def get_layout(self, region: int) -> str:
        """
        Get the tile layout specification for a region.

        Returns the tile layout string that describes how tiles are arranged
        spatially. Uses either the explicitly provided layout or derives it
        from the dataset metadata.

        Args:
            region (int): Region number to get layout for

        Returns:
            str: Layout specification string (e.g., "snake")
        """
        if self.layout:
            return self.layout
        return self.meta.layout(region)

    def create_cmd(self, region: int) -> str:
        """
        Create Ashlar command string for a specific region.

        Generates the complete Ashlar command with all necessary parameters
        including fileseries specifications, output paths, and alignment settings.
        Each cycle is added as a separate fileseries input.

        Args:
            region (int): Region number to create command for

        Returns:
            str: Complete Ashlar command string ready for execution
        """
        reg_out_dir = self.out_dir / f"reg{region:03d}"
        reg_out_dir.mkdir(exist_ok=True, parents=True)
        df_reg = self.df[self.df["region"] == region]

        str_cmd = "ashlar "
        for cycle, path in zip(df_reg["cycle"].to_list(), df_reg["path"].to_list()):
            str_cmd += (
                self.FILESERIES_TEMPLATE.format(
                    path=path,
                    region=region,
                    series="{series:05}",
                    channel="{channel}",
                    width=self.meta.width_tiles(region),
                    height=self.meta.height_tiles(region),
                    overlap=self.meta.tile_overlap,
                    layout=self.get_layout(region),
                    pixel_size=self.meta.resolution_nm / 1000.0,
                )
                + " "
            )

        str_cmd += (
            "-o "
            + osp.join(
                str(reg_out_dir),
                (
                    CodexFiles.STITCHED_TIF_FMT.format(region=region, cycle="{cycle:03}", channel="{channel:03}")
                    if not self.ome
                    else CodexFiles.STITCHED_OME_TIF
                ),
            )
            + f" --align-channel={self.align_channel} --maximum-shift={self.max_shift} --filter-sigma={self.filter_sigma}"
        )

        return str_cmd

    def reformat_codex_proc(self):
        """
        Rename Ashlar output files to match CODEX processor format.

        Converts the default Ashlar output filenames to the naming convention
        used by the CODEX processor software. This includes adding marker names
        and adjusting the numbering scheme to match CODEX expectations.
        """
        file_list = glob(str(self.out_dir / "reg*/*"))
        if len(file_list) == 0:
            raise FileNotFoundError(f"Ashlar files not found in: {self.out_dir}")

        pattern = re.compile(CodexFiles.STITCHED_TIF_RE)
        for f in file_list:
            if osp.isdir(f):
                continue

            match = pattern.match(osp.basename(f))
            if match:
                reg, cyc, ch = match.groups()
                new_cyc = int(cyc) + 1
                new_ch = int(ch) + 1

                marker = self.meta.marker_name(new_cyc, new_ch)

                new_f = osp.join(
                    osp.dirname(f),
                    CodexFiles.STITCHED_CODEX_PROC_TIF_FMT.format(region=int(reg), cycle=f"{new_cyc:03}", channel=f"{new_ch:03}", marker=marker),
                )

                shutil.move(f, new_f)
                log.info(f"Renamed: {f} -> {new_f}")

    def reformat_raw_fmt(self):
        """
        Reformat Ashlar output to CODEX raw directory structure.

        Converts Ashlar output files to the standard CODEX raw format directory
        structure. This involves creating cycle/region directories and renaming
        files to match the raw format naming convention. Also cleans up the
        original Ashlar output directories.

        This format is compatible with downstream CODEX processing tools.
        """
        file_list = glob(str(self.out_dir / "reg*/*"))
        if len(file_list) == 0:
            raise FileNotFoundError(f"Ashlar files not found in: {self.out_dir}")

        pattern = re.compile(CodexFiles.STITCHED_TIF_RE)
        for f in file_list:
            if osp.isdir(f):
                continue

            match = pattern.match(osp.basename(f))
            if match:
                reg, cyc, ch = match.groups()
                new_cyc = int(cyc) + 1
                new_ch = int(ch) + 1
                reg = int(reg)

                new_f = osp.join(
                    self.out_dir,
                    CodexFiles.RAW_DIR_FMT.format(cycle=new_cyc, region=reg),
                    CodexFiles.RAW_IMG_FMT.format(region=reg, tile=1, zslice=1, channel=new_ch),
                )

                os.makedirs(osp.dirname(new_f), exist_ok=True)
                shutil.move(f, new_f)
                log.info(f"Moved: {f} -> {new_f}")

        for f in file_list:
            base_f = osp.dirname(f)
            if osp.exists(base_f):
                os.rmdir(base_f)
                log.info(f"Removed: {base_f}")


class M2Stitch(Stitching):
    """
    Stitch tiles from a CODEx acquisition into a single 2D mosaic using a
    precomputed geometric model derived from one alignment channel/cycle.

    This component filters the provided dataset to the specified
    ``align_cycle``/``align_channel``, builds a stitching model (tile
    positions) via :func:`stitch_images`, and then places tiles from an input
    stack accordingly.

    Args:
        ds: CODEx dataset containing tiled images and metadata. The dataset is
            deep-copied internally to avoid side effects.
        out_dir: Output directory used by the base :class:`Stitching` class.
        align_cycle: Cycle index used to estimate the stitching model.
        align_channel: Channel index used to estimate the stitching model.
        max_cores: Maximum number of CPU cores for model computation.
        use_gpu: Whether to allow GPU acceleration for model computation.
        **kwargs: Additional keyword arguments forwarded to
            :func:`stitch_images` (e.g., optimizer parameters).
    """

    def __init__(
        self,
        ds: CodexDataset,
        out_dir: str | Path,
        align_cycle: int,
        align_channel: int,
        max_cores: int,
        use_gpu: bool,
        **kwargs,
    ):
        super().__init__(out_dir, is_cmd=False)
        self.ds = deepcopy(ds)
        self.meta = ds.meta

        self.ds.filter({"cycle": align_cycle, "channel": align_channel})
        self.ds.group_tiles()
        assert len(self.ds) == 1, f"Expected single image for stitching model, got {len(self.ds)}."

        self.max_cores = max_cores
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self.model_df = None

    def __call__(self, stack: np.ndarray, **kwargs) -> np.ndarray:
        """
        Stitch a 3D stack of tiles into a 2D mosaic using the cached/derived model.

        Args:
            stack: Tile stack with shape ``(tiles, height, width)``. Tile order
                must correspond to the tile ordering used when the model was
                computed (see :meth:`get_layout_indices`).
        Returns:
            A 2D numpy array containing the stitched mosaic with shape
            ``(mosaic_height, mosaic_width)``.
        """
        assert stack.ndim == 3, "Input stack must be a 3D array (tiles, height, width)."

        if self.model_df is None:
            self.compute_model()

        self.model_df["y_pos2"] = self.model_df["y_pos"] - self.model_df["y_pos"].min()
        self.model_df["x_pos2"] = self.model_df["x_pos"] - self.model_df["x_pos"].min()

        size_y = stack.shape[1]
        size_x = stack.shape[2]

        stitched_image_size = (
            self.model_df["y_pos2"].max() + size_y,
            self.model_df["x_pos2"].max() + size_x,
        )
        stitched_image = np.zeros_like(stack, shape=stitched_image_size)
        for i, row in self.model_df.iterrows():
            stitched_image[
                row["y_pos2"] : row["y_pos2"] + size_y,
                row["x_pos2"] : row["x_pos2"] + size_x,
            ] = stack[i]

        return stitched_image

    def compute_model(self):
        """
        Compute and cache the stitching model (tile positions) for the current dataset.
        """
        region = self.ds.get_unique_regions().item()
        rows, cols = self.get_layout_indices(region)
        log.info(f"Computing stitching model for region {region}")

        img = self.ds[0]["img"].compute() if self.ds.is_lazy() else self.ds[0]["img"]

        self.model_df, _ = stitch_images(img, rows, cols, max_cores=self.max_cores, use_gpu=self.use_gpu, **self.kwargs)
        self.model_df.to_pickle(self.out_dir / f"model_reg{region:03d}.pkl")

    def get_layout_indices(self, region: int) -> Tuple[List[int], List[int]]:
        """
        Generate tile row/column indices for the given region according to layout.

        For a "snake" layout, even-numbered rows increment left→right while
        odd-numbered rows decrement right→left. For a "raster" layout, all rows
        increment left→right.

        Args:
            region: Region identifier used to query layout metadata.

        Returns:
            A tuple ``(rows, cols)`` where each is a list of tile indices aligned
            with the dataset's tile ordering.

        """
        n = self.meta.height_tiles(region)  # Number of rows (height)
        m = self.meta.width_tiles(region)  # Number of columns (width)

        if self.meta.layout(region) == "snake":
            rows = list(chain.from_iterable(repeat(row, m) for row in range(n)))
            cols = list(chain.from_iterable(range(m) if row % 2 == 0 else range(m - 1, -1, -1) for row in range(n)))
        elif self.meta.layout(region) == "raster":
            rows = list(chain.from_iterable(repeat(row, m) for row in range(n)))
            cols = list(chain.from_iterable(range(m) for _ in range(n)))
        else:
            raise ValueError(f"Unknown layout: {self.meta.layout(region)}")

        return rows, cols
