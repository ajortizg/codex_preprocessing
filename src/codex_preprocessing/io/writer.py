from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile

from codex_preprocessing import CodexFiles
from codex_preprocessing.utils import ensure_path


def _create_raw_subdir(out_dir: Path | str, cycle: int, region: int) -> Path:
    """
    Create the raw subdirectory for a specific cycle and region.

    Args:
        out_dir (Path | str): Base output directory.
        cycle (int): Cycle number.
        region (int): Region identifier.

    Returns:
        Path: Path to the created raw subdirectory.
    """
    img_dir = ensure_path(out_dir) / CodexFiles.RAW_DIR_FMT.format(cycle=cycle, region=region)
    img_dir.mkdir(exist_ok=True, parents=True)
    return img_dir


def save_raw_zstack(
    out_dir: Path | str, img3d: np.ndarray, region: int, cycle: int, tile: int, channel: int, z_indices: Optional[Sequence[int]] = None
):
    """
    Save a 3D z-stack as individual z-slice TIFF files.

    Takes a 3D image stack and saves each z-slice as a separate TIFF file
    following CODEX raw data naming conventions.

    Args:
        out_dir (Path | str): Output directory for saving z-slice files.
        img3d (np.ndarray): 3D image array with shape (z_slices, height, width).
        region (int): Region identifier.
        cycle (int): Cycle number.
        tile (int): Tile number.
        channel (int): Channel number.
        z_indices Optional[Sequence[int]]: Specific z-slice indices to save.
    """
    img_dir = _create_raw_subdir(out_dir, cycle, region)

    nz, h, w = img3d.shape

    for z in range(nz):
        img = img3d[z]
        tifffile.imwrite(
            img_dir / CodexFiles.RAW_IMG_FMT.format(region=region, tile=tile, zslice=(z + 1) if z_indices is None else z_indices[z], channel=channel),
            img,
        )


def save_raw_tile(out_dir: Path | str, tile: np.ndarray, tile_num: int, region: int):
    """
    Save a complete tile containing multiple cycles and channels.

    Processes a 5D tile array and saves all cycles, channels, and z-slices
    as individual TIFF files using CODEX naming conventions.

    Args:
        out_dir (Path | str): Output directory for saving tile data.
        tile (np.ndarray): 5D array with shape (cycles, channels, z_slices, height, width).
        tile_num (int): Tile number identifier.
        region (int): Region identifier.
    """
    ncycles, nchannels, *_ = tile.shape

    for cyc in range(ncycles):
        for ch in range(nchannels):
            save_raw_zstack(out_dir, tile[cyc, ch], region, cyc + 1, tile_num, ch + 1)


def save_raw_img(out_dir: str | Path, img, region: int, cycle: int, channel: int, tile: int, zslice: int):
    """
    Save a raw image to disk in raw CODEX format.

    Args:
        out_dir (str | Path): Output directory for saving the image.
        img (np.ndarray): 2D image array to save.
    """
    assert img.ndim == 2, f"Expected a 2D array, but got {img.ndim}D array"

    img_dir = _create_raw_subdir(out_dir, cycle, region)
    tifffile.imwrite(img_dir / CodexFiles.RAW_IMG_FMT.format(region=region, tile=tile, zslice=zslice, channel=channel), img)


def save_raw_cycle_stack(out_dir: Path | str, x: np.ndarray, region: int, cycle: Optional[Sequence[int]], channel: int, tile: int, zslice: int):
    assert x.ndim == 3, f"Expected a 3D array, but got {x.ndim}D array"
    ncycles, h, w = x.shape

    for c in range(ncycles):
        save_raw_img(out_dir, x[c], region=region, cycle=(c + 1) if cycle is None else cycle[c], channel=channel, tile=tile, zslice=zslice)


def save_raw_tile_stack(
    out_dir: Path | str,
    stack: np.ndarray,
    region: int,
    cycle: int,
    channel: int,
    zslice: int,
    tile_indices: Optional[Sequence[int]] = None,
):
    assert stack.ndim == 3, f"Expected a 3D array, but got {stack.ndim}D array"

    img_dir = _create_raw_subdir(out_dir, cycle, region)
    nt, h, w = stack.shape

    for t in range(nt):
        tifffile.imwrite(
            img_dir
            / CodexFiles.RAW_IMG_FMT.format(region=region, tile=(t + 1) if tile_indices is None else tile_indices[t], zslice=zslice, channel=channel),
            stack[t],
        )
