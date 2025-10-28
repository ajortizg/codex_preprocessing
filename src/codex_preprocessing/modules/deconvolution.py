import abc
import logging
import os
import re
from typing import Any, Dict, List

import numpy as np
from flowdec import data as fd_data
from flowdec import psf as fd_psf
from flowdec import restoration as fd_restoration
from scipy import ndimage as ndi
from skimage.exposure import rescale_intensity

from codex_preprocessing.utils import clip_and_cast_to_uint, configure_tensorflow_gpus

log = logging.getLogger(__name__)


class Deconvolution(abc.ABC):
    """
    Abstract base class for CODEX image deconvolution algorithms.

    This class defines the interface for deconvolution methods used in the CODEX
    preprocessing pipeline. All concrete deconvolution algorithms should inherit
    from this class and implement the __call__ method.
    """

    def __init__(self):
        super().__init__()
        configure_tensorflow_gpus(True)

    @abc.abstractmethod
    def __call__(self, img3d: np.ndarray, channel: int) -> np.ndarray:
        """
        Apply deconvolution to a 3D image stack.

        Args:
            img3d (np.ndarray): Input 3D image stack with shape (z, y, x)
            channel (int): Channel number (1-indexed) for PSF selection

        Returns:
            np.ndarray: Deconvolved image stack with same shape as input

        Raises:
            NotImplementedError: If not implemented by concrete subclass
        """
        pass

    def generate_psfs(
        self,
        size_z: int,
        size_y: int,
        size_x: int,
        mag: float,
        na: float,
        res_axial_um: float,
        res_lateral_nm: float,
        immersion: str,
        wavelength_nm: List[float],
    ):
        """
        Generate point spread functions (PSFs) for deconvolution based on imaging parameters.

        Creates Gibson-Lanni PSF models that accurately represent the optical
        characteristics of the CODEX imaging system. PSFs are generated for each
        channel based on emission wavelengths and microscope specifications.

        Args:
            size_z (int): Number of axial planes (z-dimension) in the PSF
            size_y (int): Height of the PSF in pixels
            size_x (int): Width of the PSF in pixels
            mag (float): Microscope magnification (e.g., 20.0 for 20x objective)
            na (float): Numerical aperture of the objective lens (e.g., 0.75)
            res_axial_um (float): Axial (z) resolution in micrometers (e.g., 0.3)
            res_lateral_nm (float): Lateral (xy) resolution in nanometers (e.g., 325)
            immersion (str): Immersion medium type ("air", "water", or "oil")
            wavelength_nm (List[float]): Emission wavelengths in nanometers for each channel

        Sets:
            self.psfs (List[np.ndarray]): List of PSF arrays, one per channel
        """
        args = dict(
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            m=mag,
            na=na,
            ti0=610,
            ng0=1.5,
            tg0=170,
            res_axial=res_axial_um / 1000.0,  # in microns, TODO: is there an error with metadata file?
            res_lateral=res_lateral_nm / 1000.0,  # convert to microns
            ni0=self.get_immersion_ri(immersion.lower()),
        )
        self.psfs = [fd_psf.GibsonLanni(**{**args, **{"wavelength": w / 1000.0}}).generate() for w in wavelength_nm]

    def get_immersion_ri(self, immersion: str) -> float:
        """
        Get refractive index for an immersion medium type.

        Returns the refractive index value used in PSF calculations for different
        immersion media commonly used in microscopy. The refractive index affects
        the shape and characteristics of the point spread function.

        Args:
            immersion (str): Immersion medium type (case-insensitive)
                - "air": Standard air objective
                - "water": Water immersion objective
                - "oil": Oil immersion objective

        Returns:
            float: Refractive index value for the specified medium
        """
        if immersion == "air":
            return 1.0
        elif immersion == "water":
            return 1.33
        elif immersion == "oil":
            return 1.5115
        else:
            raise ValueError('Immersion "{}" is not valid (must be air, water, or oil)'.format(immersion))


class RLDeconvolution(Deconvolution):
    def __init__(self, n_iter: int, scale_factor: float, device: int, use_gpu: bool, **kwargs):
        """
        Initialize Richardson-Lucy deconvolution algorithm.

        Sets up the Richardson-Lucy deconvolution pipeline with specified iteration
        count and intensity scaling parameters. The algorithm uses flowdec backend
        for GPU-accelerated deconvolution and automatically configures TensorFlow
        for optimal GPU utilization.

        Args:
            n_iter (int): Number of Richardson-Lucy iterations to perform.
                Higher values (20-50) provide better deconvolution quality but
                increase computation time. Typical range: 10-50 iterations.
            scale_factor (float): Multiplicative factor for intensity rescaling.
                Applied after mean intensity restoration to fine-tune brightness:
                - 1.0: Preserve original intensity scaling (recommended)
                - > 1.0: Increase brightness of deconvolved result
                - < 1.0: Decrease brightness of deconvolved result
                - 0: Disable intensity rescaling entirely
            device (int): GPU device ID (e.g., 0 for /gpu:0)
            use_gpu (bool): Whether to use GPU acceleration (recommended)
            **kwargs: Additional keyword arguments passed to RichardsonLucyDeconvolver.
        """
        super().__init__()
        self.n_iter = n_iter
        self.scale_factor = scale_factor
        self.psfs = None
        self.algo = fd_restoration.RichardsonLucyDeconvolver(
            n_dims=3,
            device=f"/gpu:{device}" if use_gpu else f"/cpu:{device}",
            **kwargs,
        ).initialize()

    def __call__(self, img3d: np.ndarray, channel: int) -> np.ndarray:
        """
        Apply Richardson-Lucy deconvolution to a 3D image stack.

        Performs iterative deconvolution using the Richardson-Lucy algorithm
        with the appropriate PSF for the specified channel. The method handles
        intensity rescaling and data type preservation.

        Args:
            img3d (np.ndarray): Input 3D image stack with shape (z, height, width).
                Must be unsigned integer type (uint8, uint16, etc.).
            channel (int): Channel number (1-indexed) for PSF selection.
                Must correspond to a PSF generated by generate_psfs().

        Returns:
            np.ndarray: Deconvolved image stack with same shape and dtype as input.
                Intensities are rescaled and clipped to preserve quantitative values.
        """
        if not np.issubdtype(img3d.dtype, np.unsignedinteger):
            raise ValueError("Only unsigned integer images supported; type given = {}".format(img3d.dtype))
        if self.psfs is None:
            raise ValueError("PSFs must be provided or computed first")

        acq = fd_data.Acquisition(img3d, kernel=self.psfs[channel - 1])
        res = self.algo.run(acq, self.n_iter).data

        if self.scale_factor > 0:
            res, mean_ratio = self.rescale_stack(acq.data, res)

        # Clip float32 and convert to type of original image (i.e. w/ no scaling)
        dtype = acq.data.dtype
        dinfo = np.iinfo(dtype)
        return np.clip(res, dinfo.min, dinfo.max).astype(dtype)

    def rescale_stack(self, img: np.ndarray, stack: np.ndarray):
        """
        Restore mean intensity of deconvolved z-stack to preserve quantitative values.

        Richardson-Lucy deconvolution can alter the overall intensity distribution
        of images. This method rescales the deconvolved result to match the mean
        intensity of the original image, preserving quantitative measurements while
        allowing fine-tuning through the scale_factor parameter.

        This approach is based on the Nolanlab deconvolution pipeline and ensures
        that protein quantification remains consistent before and after deconvolution.

        Args:
            img (np.ndarray): Original input image stack
            stack (np.ndarray): Deconvolved image stack (float32)

        Returns:
            tuple: (rescaled_stack, mean_ratio) where:
                - rescaled_stack (np.ndarray): Intensity-corrected deconvolved stack
                - mean_ratio (float): Ratio of original to deconvolved mean intensities
        """
        mean_ratio = img.mean() / clip_and_cast_to_uint(stack, img.dtype).mean()
        log.info("Mean ratio of original stack to deconvolved stack = {}".format(mean_ratio))
        return stack * (mean_ratio * self.scale_factor), mean_ratio
