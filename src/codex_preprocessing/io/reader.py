import os.path as osp
import re
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd

from codex_preprocessing._constants import CodexFiles
from codex_preprocessing.utils import ensure_path


def read_raw_data(root_dir: Path | str, pattern: str = CodexFiles.RAW_DATA_RE) -> pd.DataFrame:
    """
    Read raw CODEX data files and extract metadata from filenames.

    Scans directories for raw CODEX image files matching the specified pattern
    and extracts cycle, region, tile, channel, and z-slice information.

    Args:
        root_dir (Path | str): Root directory containing CODEX raw data.
        pattern: Regular expression pattern for matching filenames.
            Defaults to CodexFiles.RAW_DATA_RE.

    Returns:
        pd.DataFrame: DataFrame containing metadata extracted from filenames
            with columns for cycle, region, tile, channel, zslice, and img_path.
    """
    file_list = glob(osp.join(str(root_dir), "cyc*_reg*/*"))
    if len(file_list) == 0:
        raise FileNotFoundError(f"Raw files not found in: {root_dir}")

    records = []
    for f in file_list:
        if osp.isdir(f):
            continue
        m = re.search(pattern, f)
        if m is None:
            continue

        d = {k: int(v) for k, v in m.groupdict().items()}
        d["img_path"] = f
        records.append(d)

    return pd.DataFrame.from_records(records)


def read_processed_data(root_dir: Path | str, meta: Optional[object] = None) -> pd.DataFrame:
    """
    Read processed CODEX data files from Akoya Processor output.

    Scans directories for processed image files and extracts metadata from filenames.
    Converts spatial coordinates (x, y) to tile indices using metadata information
    and returns a structured DataFrame for downstream analysis.

    Args:
        root_dir (Path | str): Root directory containing processed CODEX data files.
            Expected structure: processed*/tiles/*/*
        meta (Metadata, optional): Metadata object containing tile indexing information.
            If None, creates a new Metadata object from root_dir.

    Returns:
        pd.DataFrame: DataFrame containing extracted metadata with columns for
            region, cycle, channel, tile identifiers and image file paths.
            Spatial coordinates (x, y) are converted to tile indices and removed.
    """
    from codex_preprocessing.data import Metadata

    root_dir = ensure_path(root_dir)

    if meta is None:
        meta = Metadata(root_dir)

    file_list = glob(osp.join(str(root_dir), "processed*/tiles/*/*"))
    if len(file_list) == 0:
        raise FileNotFoundError(f"Processed files not found in: {root_dir}")

    records = []
    tile_indices = meta.tile_indices(1)  # TODO! modify this for multiple regions
    for f in file_list:
        if osp.isdir(f):
            continue
        m = re.search(CodexFiles.PROC_IMG_RE, f)
        if m is None:
            continue

        d = {k: int(v) for k, v in m.groupdict().items()}
        # region_index = int(d["region"])
        # d["tile"] = get_tile_index_from_xy(region_index, root, d["x"], d["y"])

        # d["tile"] = meta.tiling.index((d["y"], d["x"]))
        d["tile"] = tile_indices[d["x"] - 1, d["y"] - 1]  # the pos in the metadata file are 0-based
        d["img_path"] = f
        records.append(d)

    df = pd.DataFrame.from_records(records)
    df = df.drop(columns=["x", "y"])
    return df
