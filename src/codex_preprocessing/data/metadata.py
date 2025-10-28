import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from codex_preprocessing._constants import CodexFiles
from codex_preprocessing.utils import ensure_path


def read_metadata(root_dir: Path | str) -> Dict[str, Any]:
    root_dir = ensure_path(root_dir)
    filepath = root_dir / CodexFiles.META_JSON_V4_FILE
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find {filepath}")

    with filepath.open(mode="r") as f:
        experiment = json.load(f)
    return experiment


def save_metadata(root_dir: Path | str, meta: Dict[str, Any]):
    filepath = ensure_path(root_dir) / CodexFiles.META_JSON_V4_FILE

    with filepath.open(mode="w") as f:
        json.dump(meta, f, indent=2)


class Metadata:
    def __init__(self, root_dir: Path | str, meta_dict: Optional[Dict[str, Any]] = None):
        if meta_dict is not None:
            self.meta = meta_dict
        else:
            self.meta = read_metadata(root_dir)

        # Cycle metadata
        self.cycle_channel_map = {cycle["index"]: {channel["index"]: channel for channel in cycle["channels"]} for cycle in self.meta["cycles"]}

        # Read region metadata
        self.regions = {
            region["index"]: {
                "region_info": {k: v for k, v in region.items() if k != "tiles"},  # Store region metadata
                "tiles": {tile["index"]: tile for tile in region["tiles"]},  # Store tile data
            }
            for region in self.meta["regions"]
        }

        # To avoid unecessary calculations
        self.layout_per_region = {}

    def get_dict(self):
        return self.meta

    def marker_name(self, cycle: int, channel: int) -> str:
        "Expects 1-based indices"
        return self.cycle_channel_map[cycle][channel]["markerName"].replace("/", "-").strip()

    @property
    def n_channels(self) -> int:
        # All cycles must have the same number of channels, lets take the first one
        cycle = 1
        return len(self.cycle_channel_map[cycle])

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def slide_id(self) -> str:
        return self.meta["slideId"]

    @property
    def n_cycles(self) -> int:
        return len(self.meta["cycles"])

    @property
    def n_zplanes(self) -> int:
        return self.meta["numPlanes"]

    def save(self, out_dir: Path | str):
        save_metadata(out_dir, self.meta)

    @property
    def bit_depth(self) -> int:
        return self.meta["microscopeBitDepth"]

    @property
    def dtype(self):
        d = self.bit_depth
        if d == 16:
            return np.uint16
        elif d == 8:
            return np.uint8
        else:
            raise ValueError(f"Unknow microscopeBitDepth: {d}")

    @property
    def immersion(self) -> str:
        im: str = self.meta["objective"]["immersion"]
        return im.lower()

    @property
    def magnification(self) -> float:
        return self.meta["objective"]["magnification"]

    @property
    def aperture(self) -> float:
        return self.meta["objective"]["numericalAperture"]

    @property
    def resolution_nm(self) -> float:
        return self.meta["resolution_nm"]

    @property
    def pitch_um(self) -> float:
        return self.meta["pitch_um"]

    @property
    def tile_overlap(self) -> float:
        return self.meta["tileOverlap"]

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def width_tiles(self, region: int) -> int:
        return self.regions[region]["region_info"]["regionWidth_tiles"]

    def height_tiles(self, region: int) -> int:
        return self.regions[region]["region_info"]["regionHeight_tiles"]

    def n_tiles(self, region: int) -> int:
        return len(self.regions[region]["tiles"])

    def tile_pos(self, region: int, index: int) -> Tuple[int, int]:
        "Return 0-based coordinates"
        x = self.regions[region]["tiles"][index]["x"]
        y = self.regions[region]["tiles"][index]["y"]
        return x, y

    def tile_indices(self, region: int) -> Dict[Tuple[int, int], int]:
        "x,y 0-based coordinates."
        return {(tile_data["x"], tile_data["y"]): tile_index for tile_index, tile_data in self.regions[region]["tiles"].items()}

    def layout(self, region: int) -> str:
        """
        Infer tile scanning layout for a region ("raster" or "snake").

        Determines whether tiles in the given region were acquired using a
        raster pattern (left-to-right on every row) or a serpentine "snake"
        pattern (alternating direction every other row). The inference is
        performed by comparing the x-indices from metadata with the x-indices
        expected from a pure raster acquisition.

        Method:
        - Build X_raster: expected x-index sequence for a raster scan given the
            region width and height (0-based, repeated per row).
        - Build X_meta: x-indices extracted from tile metadata (0-based).
        - If X_meta equals X_raster for all tiles, return "raster"; otherwise
            return "snake".

        Results are cached per region in `self.layout_per_region` to avoid
        recomputation on subsequent calls.

        Args:
                region (int): Region identifier as stored in metadata.

        Returns:
                str: "raster" if tiles follow left-to-right order on every row;
                            "snake" if rows alternate direction.
        """
        if region in self.layout_per_region:
            return self.layout_per_region[region]
        else:
            X_meta = np.zeros(self.n_tiles(region), dtype=np.uint)
            X_raster = np.tile(np.arange(self.width_tiles(region)), (self.height_tiles(region), 1)).flatten().astype(np.uint)

            for i in range(self.n_tiles(region)):
                x, _ = self.tile_pos(region, i + 1)
                X_meta[i] = x

            lay = "raster" if np.all(X_meta == X_raster) else "snake"
            self.layout_per_region[region] = lay
            return lay

    @property
    def wavelength_nm(self) -> List[float]:
        # must be the same for all cycles, lets use cycle 1
        cycle = 1
        wv = [None] * self.n_channels
        for ch_index, ch_data in self.cycle_channel_map[cycle].items():
            wv[ch_index - 1] = ch_data["wavelength_nm"]
        return wv

    def set_zplanes(self, zplanes: int):
        self.meta["numPlanes"] = zplanes
