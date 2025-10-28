import abc
import logging
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Tuple

import dask.array as da
import numpy as np
import pandas as pd

from codex_preprocessing.data import CodexDataset
from codex_preprocessing.io import save_raw_cycle_stack, save_raw_img

log = logging.getLogger(__name__)


class BackgroundCorrector(abc.ABC):
    """
    Base class for background correction algorithms.

    Background correctors remove autofluorescence and imaging artifacts
    from CODEX images to improve signal quality and quantification accuracy.

    Args:
        ds: CODEX dataset for extracting blank cycle references.
        first_cycle: 1-based first cycle index (typically a blank).
    """

    def __init__(self, ds: CodexDataset, first_cycle: int = 1):
        super().__init__()
        self.meta = ds.meta
        self.first_cycle = first_cycle
        self.last_cycle = self.meta.n_cycles

    @abc.abstractmethod
    def __call__(self, x: da.Array | np.ndarray, cycle: int, channel: int) -> np.ndarray:
        """
        Apply background correction to an image.

        Args:
            x: Input image (2D or 3D array).
            cycle: 1-based cycle index.
            channel: 1-based channel index.

        Returns:
            Background-corrected image.
        """
        pass

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        return ds

    @abc.abstractmethod
    def save(self, out_dir: str | Path, img: np.ndarray, region, cycle, channel, tile, zslice):
        """
        Save corrected image to disk.

        Args:
            out_dir: Output directory path.
            img: Corrected image array.
            region: Region identifier.
            cycle: Cycle number(s).
            channel: Channel number.
            tile: Tile number.
            zslice: Z-slice number.
        """
        pass

    def _prepare_data(self, ds: CodexDataset):
        ds = deepcopy(ds)
        ds.set_lazy()
        ds.sort_by("cycle")
        ds.group_channels()
        return ds

    def get_blanks(self, ds: CodexDataset) -> Tuple[da.Array, da.Array]:
        ds = self._prepare_data(ds)
        bi = ds[self.first_cycle - 1]["img"]
        bf = ds[self.last_cycle - 1]["img"]
        return bi, bf


class LinearInterpolationCorrector(BackgroundCorrector):
    """
    Background correction using linear interpolation between blank cycles.

    Estimates background for each cycle by interpolating between the first
    and last cycle blank images. Subtracts the interpolated background to
    reduce autofluorescence and imaging artifacts.

    Args:
        ds: CODEX dataset containing all cycles.
        nuclear_channel: 1-based nuclear marker channel (excluded from correction).
        skip_markers: Marker names to skip (e.g., blanks or controls).
        first_cycle: 1-based first cycle for initial blank reference.
    """

    def __init__(
        self,
        ds: CodexDataset,
        nuclear_channel: int,
        skip_markers: Sequence[str] = [],
        first_cycle: int = 1,
    ):
        super().__init__(ds, first_cycle)
        self.nuclear_channel = nuclear_channel
        self.skip_markers = set(skip_markers)

        self.bis, self.bfs = self.get_blanks(ds)
        self.scaling_factors = np.linspace(0, 1, self.meta.n_cycles - 2)
        self.scaling_factors = np.pad(self.scaling_factors, (1, 1), mode="edge")

    def __call__(self, x: da.Array | np.ndarray, cycle: int, channel: int) -> np.ndarray:
        marker = self.meta.marker_name(cycle, channel)
        x = x.compute() if isinstance(x, da.Array) else x
        x = x.squeeze(0) if x.ndim == 3 else x
        assert x.ndim == 2, "Input image must be 2D (height, width)"

        if cycle == self.first_cycle or cycle == self.last_cycle or channel == self.nuclear_channel or marker in self.skip_markers:
            return x

        dtype = x.dtype
        dinfo = np.iinfo(dtype)

        b = (
            self.bis[channel - 1].compute().astype(np.float32) * (1.0 - self.scaling_factors[cycle - 1])
            + self.bfs[channel - 1].compute().astype(np.float32) * self.scaling_factors[cycle - 1]
        )
        out = np.clip(x.astype(np.float32) - b, dinfo.min, dinfo.max).astype(dtype)
        return out

    def save(self, out_dir: str | Path, img: np.ndarray, region, cycle, channel, tile, zslice):
        save_raw_img(out_dir, img, region, cycle, channel, tile, zslice)


class AutofluorescenceCorrector(BackgroundCorrector):
    """
    Adaptive autofluorescence correction using probe-based intensity modeling.

    Identifies high-intensity probe pixels from blank cycles and tracks their
    intensity across all cycles to compute adaptive correction factors. Each
    cycle is corrected using its specific scaling coefficient.

    Args:
        ds: CODEX dataset containing all cycles.
        skip_markers: Marker names to skip correction.
        q1: Lower quantile threshold for probe selection (0-1).
        q2: Upper quantile threshold for probe selection (0-1).
        nuclear_channel: 1-based nuclear marker channel (excluded from correction).
        first_cycle: 1-based first cycle for blank reference.
    """

    def __init__(
        self,
        ds: CodexDataset,
        skip_markers: Sequence[str] = [],
        q1: float = 0.9,
        q2: float = 1.0,
        nuclear_channel: int = 1,
        first_cycle: int = 1,
    ):
        super().__init__(ds, first_cycle)
        self.skip_markers = set(skip_markers)
        self.q1 = q1
        self.q2 = q2
        self.nuclear_channel = nuclear_channel

        self.bis, self.bfs = self.get_blanks(ds)

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        ds.group_cycles()
        return ds

    def get_probes(self, B):
        Q1 = np.quantile(B, self.q1)
        Q2 = np.quantile(B, self.q2)
        return np.transpose(np.nonzero((B <= Q2) & (B > Q1)))

    def estimate_autofluorescence(self, img, af_image, cycles):
        probes = self.get_probes(af_image)
        num_probes = len(probes)
        num_cycles = len(img)

        probe_values = np.zeros((num_cycles, num_probes))

        af_probe_value = af_image[probes[:, 0], probes[:, 1]]

        # Extract probe values for each cycle
        for i, img_cyc in enumerate(img):
            probe_values[i] = img_cyc.compute()[probes[:, 0], probes[:, 1]]

        # Create DataFrame for cycle-wise probe values
        df = pd.DataFrame(
            {
                "point_index": np.tile(np.arange(num_probes), num_cycles),
                "cycle": np.repeat(cycles, num_probes),
                "value": probe_values.ravel(),
                "q1": np.full(num_probes * num_cycles, self.q1),
                "q2": np.full(num_probes * num_cycles, self.q2),
            }
        )

        return df, af_probe_value, probes

    def calc_correction(self, probe_df, af_values):
        cycle_mean = probe_df.groupby("cycle").mean().reset_index()
        cycle_mean["c"] = cycle_mean["value"] / af_values.mean()
        return cycle_mean

    def adaptive_model(self, img, af_image, c):
        dtype = img.dtype
        dinfo = np.iinfo(dtype)
        return np.clip((img / c["c"]) - af_image, dinfo.min, dinfo.max).astype(dtype)

    def __call__(self, cycle_stack: da.Array | np.ndarray, cycle: Sequence[int], channel: int) -> np.ndarray:
        assert cycle_stack.ndim == 3, "Input image must be 3D (cycle, height, width)"

        if channel == self.nuclear_channel:
            return cycle_stack.compute() if isinstance(cycle_stack, da.Array) else cycle_stack

        blanks = np.max(np.stack([self.bis[channel - 1].compute(), self.bfs[channel - 1].compute()]), axis=0)
        e1, probe, _ = self.estimate_autofluorescence(cycle_stack, blanks, cycle)
        corrections = self.calc_correction(e1, probe)

        out = []
        dtype = cycle_stack.dtype
        for cyc, img in zip(cycle, cycle_stack):
            marker = self.meta.marker_name(cyc, channel)

            if marker in self.skip_markers or cyc == self.first_cycle or cyc == self.last_cycle:
                out.append(img.compute())
            else:
                c = corrections.loc[corrections["cycle"] == cyc].iloc[0]
                out.append(self.adaptive_model(img.compute(), blanks, c))

        return np.stack(out, axis=0, dtype=dtype)

    def save(self, out_dir: str | Path, img: np.ndarray, region, cycle, channel, tile, zslice):
        save_raw_cycle_stack(out_dir, img, region, cycle, channel, tile, zslice)
