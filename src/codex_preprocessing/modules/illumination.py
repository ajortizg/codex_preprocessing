import abc
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from codex_preprocessing.modules.basicpy.basicpy import BaSiC
from codex_preprocessing.utils import ensure_path

log = logging.getLogger(__name__)


class IlluminationCorrector(abc.ABC):
    """
    Abstract base class for CODEX illumination correction algorithms.

    This class defines the interface for illumination correction methods used
    in the CODEX preprocessing pipeline. All concrete illumination correction
    algorithms should inherit from this class and implement the __call__ method.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """
        Apply illumination correction to a CODEX cycle.

        Args:
            stack (np.ndarray): Input stack array to be corrected

        Returns:
            np.ndarray: Illumination-corrected array
        """
        pass


class Basic(IlluminationCorrector):
    """
    BaSiC illumination correction for CODEX images.

    Implements the BaSiC (Background and Shading Correction) algorithm for
    correcting illumination artifacts in CODEX multiplexed imaging data.

    Args:
        eps (float): Small value to avoid division by zero.
        **kwargs: Keyword arguments for BaSiC algorithm.
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__()
        self.basic_args = kwargs
        self.eps = eps

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        assert stack.ndim == 3, f"Expected a 3D array, but got {stack.ndim}D array"

        basic = BaSiC(**self.basic_args)
        basic.fit(stack)
        self.flatfield = basic.flatfield
        self.darkfield = basic.darkfield
        self.baseline = basic.baseline

        corrected_stack = self.transform(stack)
        return corrected_stack

    def save_field_imgs(self, out_dir: str | Path, cycle: int, channel: int):
        """
        Save debug visualizations of BaSiC correction fields.

        Args:
            out_dir (Path | str): Output directory for debug visualizations and logs
            cycle (int): Cycle number for file naming (1-indexed).
            channel (int): Channel number for file naming (1-indexed).
        """

        # plot dark and flatfield
        fig, ax = plt.subplots(1, 3 if self.darkfield is not None else 2, figsize=(10, 3))
        ax[0].set_title("Flatfield")
        im = ax[0].imshow(self.flatfield)
        fig.colorbar(im, ax=ax[0])

        if self.darkfield is not None:
            ax[1].set_title("Darkfield")
            im = ax[1].imshow(self.darkfield)
            fig.colorbar(im, ax=ax[1])

        bidx = 2 if self.darkfield is not None else 1
        ax[bidx].set_title("Baseline")
        ax[bidx].plot(self.baseline)
        ax[bidx].set_xlabel("Frame")
        ax[bidx].set_ylabel("Baseline")

        fig.tight_layout()
        out_dir = ensure_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"cyc{cycle:03d}_ch{channel:03d}.png")
        plt.close(fig)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply illumination correction transformation to image data.

        Applies the estimated darkfield and flatfield corrections to normalize
        illumination in the input images. The transformation handles both
        additive (darkfield) and multiplicative (flatfield) corrections while
        preserving the original data type and valid intensity range.

        Args:
            x (np.ndarray): Input image array to be corrected with shape (tiles, h, w).

        Returns:
            np.ndarray: Corrected image array with same shape and dtype as input.
                Values are clipped to the valid range for the input data type.
        """
        dtype = x.dtype
        dinfo = np.iinfo(dtype)

        if self.darkfield is None:
            y = x.astype(np.float32) / (self.flatfield + self.eps)
        else:
            y = np.maximum(x.astype(np.float32) - self.darkfield, 0) / (self.flatfield + self.eps)

        return np.clip(y, dinfo.min, dinfo.max).astype(dtype)
