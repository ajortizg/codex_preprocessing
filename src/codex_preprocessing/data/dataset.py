import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence

import dask
import dask.array
import dask_image
import dask_image.imread
import numpy as np
import pandas as pd
from skimage.io import imread
from tabulate import tabulate

from codex_preprocessing.data.metadata import Metadata
from codex_preprocessing.io import read_processed_data, read_raw_data
from codex_preprocessing.utils import ensure_path

log = logging.getLogger(__name__)


class CodexDataset:
    """
    PyTorch Dataset for CODEX multiplexed imaging data.

    Args:
        root_dir (Path | str): Root directory containing CODEX data files.
        mode (Literal["raw", "proc"]): Data loading mode - "raw" for raw format, "proc" for CODEX processor format.
        lazy_loading (bool): Whether to use lazy loading with dask arrays.
        read_markers (bool): Whether to read marker names from metadata.
    """

    def __init__(
        self,
        root_dir: Path | str,
        mode: Literal["raw", "proc"],
        lazy_loading: bool,
        read_markers: bool = False,
    ):

        self.set_mode(mode)
        self.set_lazy(lazy_loading)
        self.set_read_markers(read_markers)
        self.load_dataset(root_dir)

    def load_dataset(self, root_dir: str | Path):
        self.root_dir = ensure_path(root_dir)
        self.meta = Metadata(self.root_dir)

        if self.mode == "raw":
            df = read_raw_data(self.root_dir)
        elif self.mode == "proc":
            df = read_processed_data(self.root_dir, self.meta)
            df["zslice"] = 1
            df["tile"] = 1
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.df = df
        self.df["img_path"] = self.df["img_path"].apply(lambda x: [x])
        self._update_fields()

    def set_mode(self, mode: Literal["raw", "proc"]):
        assert mode in {"raw", "proc"}, "mode must be either raw or proc."
        self.mode = mode

    def set_read_markers(self, read_markers: bool):
        self.read_markers = read_markers

    def set_lazy(self, lazy_loading: bool = True):
        self.lazy_loading = lazy_loading

    def read_data(self, img_paths, cyc) -> Dict[str, Any]:
        """
        Read image data from file paths.

        Args:
            img_paths: List of image file paths to read.
            cyc: Cycle number for marker name lookup.

        Returns:
            Dict[str, Any]: Dictionary containing image data and optionally markers.
        """
        if not self.is_lazy():
            img = np.stack([imread(p) for p in img_paths], axis=0)
            return {"img": img}
        else:
            img = []
            markers = []
            for ch, p in enumerate(img_paths):
                img.append(dask_image.imread.imread(p))
                if self.read_markers:
                    markers.append(self.meta.marker_name(cyc, ch + 1))
            return {
                "img": dask.array.concatenate(img),
                "markers": markers,
            }

    def filter_by_column(self, column: str | Sequence[str], keep: int | Sequence[int]):
        """
        Filter dataset by column values in-place.

        Args:
            column (str | Sequence[str]): Column name(s) to filter by.
            keep (int | Sequence[int]): Value(s) to keep for each column.
        """
        if isinstance(column, str):
            column = [column]
            keep = [keep] * len(column)
        assert len(column) == len(keep), "Column and keep must have the same length."

        for col, val in zip(column, keep):
            self.df = self.df[self.df[col] == val].reset_index(drop=True)
            self._update_fields()

    def filter(self, d: Dict[str, Any]):
        """
        Filter dataset by dictionary of column-value pairs in-place.

        Args:
            d: Dictionary mapping column names to values to keep.
        """
        for col, val in d.items():
            self.df = self.df[self.df[col] == val].reset_index(drop=True)
            self._update_fields()

    def make_filtered_copy(self, column: str, keep: int) -> "CodexDataset":
        """
        Create a filtered copy of the dataset.

        Args:
            column (str): Column name to filter by.
            keep (int): Value to keep for the specified column.

        Returns:
            CodexDataset: New filtered dataset instance.
        """
        new_ds = deepcopy(self)
        # new_ds.filter_by_column(column, keep)
        new_ds.filter({column: keep})
        return new_ds

    def make_copy(self, d: Optional[Dict[str, Any]] = None) -> "CodexDataset":
        new_ds = deepcopy(self)
        if d is not None:
            new_ds.filter(d)
        return new_ds

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        img_paths = self.paths[i]
        ch = self.channels[i]
        cyc = self.cycles[i]
        tile = self.tiles[i]
        region = self.regions[i]
        zslice = self.zslices[i]

        data = self.read_data(img_paths, cyc)
        data["paths"] = img_paths
        data["channel"] = ch
        data["cycle"] = cyc
        data["tile"] = tile
        data["region"] = region
        data["zslice"] = zslice
        return data

    def get_where(self, column, value) -> pd.DataFrame:
        """
        Get rows where column equals value.

        Args:
            column: Column name to filter by.
            value: Value to match in the column.

        Returns:
            pd.DataFrame: Filtered DataFrame subset.
        """
        df_f = self.df[self.df[column] == value]
        return df_f.reset_index(drop=True)

    def group(self, column: str):
        """
        Group dataset by specified column.

        Args:
            column (str): Column name to group by.
        """
        small_df = self.df.head()
        list_columns = [col for col in small_df if small_df[col].apply(lambda x: isinstance(x, list)).any() and col != "img_path"]
        remaining_columns = [col for col in small_df.columns if col not in [column, "img_path"] + list_columns]

        agg_dict = {column: lambda x: list(x), "img_path": lambda x: sum(x, [])}
        if len(list_columns) > 0:
            agg_dict.update({col: lambda x: sum(x, []) for col in list_columns})

        df = self.df.sort_values(column).groupby(remaining_columns).agg(agg_dict).reset_index()
        self.df = df
        self._update_fields()

    def group_tiles(self):
        self.group("tile")

    def group_channels(self):
        self.group("channel")

    def group_zslices(self):
        self.group("zslice")

    def group_cycles(self):
        self.group("cycle")

    def sort_by(self, column: str | Sequence[str], **kwargs):
        self.df.sort_values(by=column, inplace=True, **kwargs)
        self.df.reset_index(drop=True, inplace=True)
        self._update_fields()

    def _update_fields(self):
        """
        Update internal field lists from the current DataFrame state.
        """
        self.paths = self.df["img_path"].tolist()
        self.channels = self.df["channel"].tolist()
        self.cycles = self.df["cycle"].tolist()
        self.tiles = self.df["tile"].tolist()
        self.regions = self.df["region"].tolist()
        self.zslices = self.df["zslice"].tolist()
        assert len(self.paths) == len(self.channels) == len(self.cycles) == len(self.tiles) == len(self.regions) == len(self.zslices)

    def get_unique_regions(self) -> np.ndarray:
        """
        Get array of unique region identifiers in the dataset.

        Returns:
            np.ndarray: Array of unique region IDs.
        """
        return self.df["region"].unique()

    def pretty_print(self, stream: Callable = print):
        """
        Print formatted table representation of the dataset.

        Args:
            stream (Callable): Function to output the table (default: print).
        """
        t = tabulate(self.df, headers="keys", tablefmt="psql")
        stream(f"{t}")

    def is_lazy(self) -> bool:
        """
        Check if the dataset is using lazy loading.

        Returns:
            bool: True if lazy loading is enabled, False otherwise.
        """
        return self.lazy_loading
