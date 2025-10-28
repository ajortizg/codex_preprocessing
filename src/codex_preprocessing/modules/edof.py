import abc
import logging
from typing import Optional

import cv2
import numpy as np
import pywt
import torch
from pytorch_wavelets import DTCWTForward

from codex_preprocessing.data import CodexDataset
from codex_preprocessing.io import save_raw_tile, save_raw_zstack

log = logging.getLogger(__name__)


class EDoF(abc.ABC):
    """
    Base class for Extended Depth of Field (EDoF) algorithms.

    EDoF methods combine multiple focal planes from a z-stack into a single
    in-focus image. Subclasses implement different fusion strategies.

    Args:
        use_gpu: Enable GPU acceleration if available.
        device: GPU device ID to use (optional).
    """

    def __init__(self, use_gpu: bool, device: Optional[int] = None):
        super().__init__()
        self.use_gpu = torch.cuda.is_available() and use_gpu
        torch.set_default_dtype(torch.float32)
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        if device is not None and self.use_gpu:
            torch.cuda.set_device(f"cuda:{device}")

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        return ds

    def preprocess_image(self, img: np.ndarray, meta=None) -> np.ndarray:
        return img

    def save(self, out_dir, img, region, cycle, tile, channel):
        save_raw_zstack(out_dir, img, region, cycle, tile, channel)

    def set_device(self, device: int):
        if self.use_gpu:
            torch.cuda.set_device(f"cuda:{device}")

    @abc.abstractmethod
    def __call__(self, img3d: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process a 3D image stack to produce an in-focus image.

        Args:
            img3d: Input z-stack with shape (z, height, width).
            **kwargs: Algorithm-specific parameters.

        Returns:
            Focused image array.
        """
        pass


class EDoFSobel(EDoF):
    """
    EDoF using Sobel edge detection for focus assessment.

    Detects focused regions using Sobel edge strength, then blends z-planes
    with Gaussian-weighted contributions based on focus quality.

    Args:
        gaussian_size: Kernel size for smoothing the focus map.
        use_gpu: Enable GPU acceleration (uses CuPy/cuCIM).
        device: GPU device ID (optional).
    """

    def __init__(self, gaussian_size: int = 5, use_gpu: bool = True, device: Optional[int] = None):
        super().__init__(use_gpu, device=device)
        self.gaussian_size = gaussian_size

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        ds.group_zslices()
        return ds

    def __call__(self, img3d: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Sobel-based EDoF to a z-stack.

        Args:
            img3d: Input stack with shape (z, height, width).
            **kwargs: Unused.

        Returns:
            Focused image with shape (1, height, width).
        """
        if self.use_gpu:
            import cupy as np
            from cucim.skimage.exposure import rescale_intensity
            from cucim.skimage.filters import gaussian, sobel
        else:
            import numpy as np
            from skimage.exposure import rescale_intensity
            from skimage.filters import gaussian, sobel

        img3d = np.array(img3d)
        nz, height, width = img3d.shape

        focus = np.array([sobel(t) for t in img3d])
        best_layer = np.argmax(focus, 0)  # best pixel along z

        # img = img3d.reshape((nz, -1))  # image is now (stack, nr_pixels)
        # img = img.transpose()  # image is now (nr_pixels, stack)
        # r = img[np.arange(len(img)), best_layer.ravel()]  # Select the right pixel at each location
        # r = r.reshape((height, width))  # reshape to get final result

        best_layer = gaussian(best_layer, self.gaussian_size, preserve_range=True)

        sigma = gaussian(np.sum(img3d / np.max(img3d), axis=0), sigma=0, preserve_range=True)
        sigma = rescale_intensity(sigma, out_range=(0, 1))

        window = lambda x, s: (np.where(np.abs(x) > 1, 0, np.where(x >= 0, 1 - x, 1 + x)))

        weigths = np.array([window(-1 * (i - best_layer), sigma) for i in range(nz)])
        result = np.nan_to_num(np.mean(img3d * weigths, axis=0))

        dtype = img3d.dtype
        dinfo = np.iinfo(dtype)
        result = np.clip(result, dinfo.min, dinfo.max).astype(dtype)

        result = np.expand_dims(result, 0)
        return result.get() if self.use_gpu else result


class EDoFWavelet(EDoF):
    """
    EDoF using Dual-Tree Complex Wavelet Transform (DT-CWT).

    Analyzes focus quality using wavelet decomposition of high-frequency content.
    Blends z-planes using exponentially normalized weights based on wavelet coefficients.
    DT-CWT provides better directional selectivity than standard wavelets.

    Args:
        biort: Biorthogonal filter for first decomposition level.
        qshift: Q-shift filter for subsequent levels.
        level: Number of wavelet decomposition levels.
        crop: Pixels to crop from result borders.
        gaussian_size: Gaussian smoothing size (currently unused).
        use_gpu: Enable GPU acceleration.
        nuclear_channel: Nuclear channel number for special handling.
        max_project_non_nuclear_channels: Use max projection for non-nuclear channels.
        device: GPU device ID (optional).
    """

    def __init__(
        self,
        biort: str = "near_sym_a",
        qshift: str = "qshift_a",
        level: int = 2,
        crop: int = 0,
        gaussian_size: int = 5,
        use_gpu: bool = True,
        nuclear_channel: int = 1,
        max_project_non_nuclear_channels: bool = True,
        device: Optional[int] = None,
    ):
        super().__init__(use_gpu, device=device)
        self.algo = DTCWTForward(J=level + 1, biort=biort, qshift=qshift)
        if device is not None:
            self.set_device(device)

        self.crop = crop
        self.gaussian_size = gaussian_size
        self.level = level
        self.nuclear_channel = nuclear_channel
        self.max_project_non_nuclear_channels = max_project_non_nuclear_channels

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        ds.group_zslices()
        return ds

    def set_device(self, device):
        super().set_device(device)
        self.algo = self.algo.to(self.device)

    def __call__(self, img3d: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply DT-CWT-based EDoF to a z-stack.

        Uses wavelet transforms to compute focus scores and blends planes
        with normalized weights. Preserves original data type.

        Args:
            img3d: Input stack with shape (z, height, width).
            **kwargs: Unused.

        Returns:
            Focused image with shape (1, height, width), cropped if specified.
        """
        if not self.use_gpu:
            import numpy as np
            import scipy.ndimage as ndi
            from skimage.filters import gaussian

            X = torch.from_numpy(img3d.astype(np.float32)).to(self.device)
        else:
            import cupyx.scipy.ndimage as ndi
            import numpy as np
            from cucim.skimage.filters import gaussian

            X = torch.from_numpy(img3d.astype(np.float32)).to(self.device)
            import cupy as np

        img3d = np.array(img3d)

        def metric(img2d):
            Yl, Yh = self.algo(img2d.unsqueeze(0).unsqueeze(0))
            score = sum(torch.sum(torch.abs(band)) for level in Yh for band in level).item()
            return score

        scores = np.array([metric(X[z]) for z in range(X.shape[0])])

        # Blends the Z-stack using a weighted sum based on focus scores.
        weights = np.exp(scores - np.max(scores))  # Normalize weights
        weights /= np.sum(weights)  # Normalize to sum to 1
        blended = np.tensordot(weights, img3d, axes=(0, 0))

        dtype = img3d.dtype
        dinfo = np.iinfo(dtype)
        blended = np.clip(blended, dinfo.min, dinfo.max).astype(dtype)

        if self.crop > 0:
            blended = blended[self.crop : -self.crop, self.crop : -self.crop]

        blended = np.expand_dims(blended, 0)
        return blended.get() if self.use_gpu else blended

        # best_focus_tile[cyc, ch, 0] = self.blend_focus_stack(img3d, scores)

        # channel = kwargs.get("channel")
        # assert channel is not None, "Channel must be provided."

        # if not self.use_gpu:
        #     import numpy as np
        #     from skimage.exposure import rescale_intensity
        #     from skimage.filters import gaussian
        #     from skimage.transform import resize

        #     X = torch.from_numpy(np.expand_dims(img3d.astype(np.float32), 1)).to(self.device)
        # else:
        #     import numpy as np
        #     from cucim.skimage.exposure import rescale_intensity
        #     from cucim.skimage.filters import gaussian
        #     from cucim.skimage.transform import resize

        #     X = torch.from_numpy(np.expand_dims(img3d.astype(np.float32), 1)).to(self.device)
        #     import cupy as np

        # img3d = np.array(img3d)

        # if channel != self.nuclear_channel and self.max_project_non_nuclear_channels:
        #     result = np.expand_dims(np.max(img3d, axis=0)[self.crop : -self.crop, self.crop : -self.crop], 0)
        #     return result.get() if self.use_gpu else result

        # Yl, Yh = self.algo(X)
        # coeff_stack = np.swapaxes(np.array(Yh[self.level][..., 0] + 1j * Yh[self.level][..., 1]), 1, 2)

        # main_focus_layer = np.argmax(np.max(np.abs(coeff_stack), axis=1), axis=0)[0]
        # main_focus_layer_large = gaussian(
        #     resize(main_focus_layer, img3d[0].shape, order=1, preserve_range=True),
        #     self.gaussian_size,
        #     preserve_range=True,
        # )

        # sigma = gaussian(np.sum(img3d / np.max(img3d), axis=0), sigma=0, preserve_range=True)
        # sigma = rescale_intensity(sigma, out_range=(0, 1))

        # # z-merging

        # window = lambda x, s: (np.where(np.abs(x) > 1, 0, np.where(x >= 0, 1 - x, 1 + x)))
        # # def window(x, s):
        # #     alpha = 0.1  # Increase contrast in weighting
        # #     return np.exp(-alpha * (x**2) / (s + 1e-6))  # Gaussian-like weight

        # weights = np.array([window(-1 * (i - main_focus_layer_large), sigma) for i in range(img3d.shape[0])])
        # # weights_sum = np.sum(weights, axis=0, keepdims=True) + 1e-6  # Prevent division by zero
        # # weights /= weights_sum  # Normalize to sum to 1

        # result = np.nan_to_num(np.mean(img3d * weights, axis=0))

        # dtype = img3d[0].dtype
        # dinfo = np.iinfo(dtype)
        # result = np.clip(result, dinfo.min, dinfo.max).astype(dtype)

        # if self.crop > 0:
        #     result = result[self.crop : -self.crop, self.crop : -self.crop]

        # result = np.expand_dims(result, 0)
        # return result.get() if self.use_gpu else result


class FocusMetric(abc.ABC):
    """
    Base class for focus quality metrics.

    Focus metrics evaluate image sharpness to determine which z-plane
    contains the best-focused content. Higher scores indicate better focus.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """
        Compute focus score for a 2D image.

        Args:
            x: Input 2D image.

        Returns:
            Focus score (higher = better focus).
        """
        pass


class WhitenNorm(FocusMetric):
    """
    Focus metric using Gaussian-Laplacian filtering with L2 norm.

    Applies Gaussian smoothing followed by Laplacian edge detection,
    then computes the L2 norm. Reduces noise while emphasizing sharp edges.

    Args:
        sigma: Standard deviation for Gaussian blur.
    """

    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> float:
        """
        Compute focus score using Gaussian-Laplacian L2 norm.

        Args:
            x: Input 2D image.

        Returns:
            L2 norm of the Laplacian-filtered image.
        """
        x = cv2.GaussianBlur(x.astype(float), (0, 0), self.sigma)
        x = cv2.Laplacian(x, cv2.CV_64F, ksize=1)
        return np.linalg.norm(x)


class LaplacianVariance(FocusMetric):
    """
    Focus metric using Laplacian variance.

    Computes variance of the Laplacian-filtered image, a classical sharpness measure.
    The Laplacian highlights edges and intensity changes.

    Args:
        ksize: Laplacian kernel size (1, 3, 5, etc.).
    """

    def __init__(self, ksize: int):
        super().__init__()
        self.ksize = ksize

    def __call__(self, x: np.ndarray) -> float:
        """
        Compute focus score using Laplacian variance.

        Args:
            x: Input 2D image.

        Returns:
            Variance of the Laplacian-filtered image.
        """
        return np.var(cv2.Laplacian(x, cv2.CV_64F, ksize=self.ksize))


class SobelVar(FocusMetric):
    """
    Focus metric using Sobel variance.

    Applies Sobel edge detection and computes variance. The Sobel operator
    detects gradients in horizontal and vertical directions.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray) -> float:
        """
        Compute focus score using Sobel variance.

        Args:
            x: Input 2D image.

        Returns:
            Variance of the Sobel-filtered image.
        """
        x = cv2.Sobel(x.astype(float), cv2.CV_64F, 1, 1, ksize=1)
        return x.var()


class WaveletMetric(FocusMetric):
    """
    Focus metric using discrete wavelet transform coefficients.

    Performs multi-level wavelet decomposition and sums absolute values of
    high-frequency detail coefficients. High-frequency content indicates
    edges, textures, and sharp details.

    Args:
        wavelet: Wavelet type for decomposition.
        level: Number of decomposition levels.
    """

    def __init__(self, wavelet, level):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def __call__(self, x):
        """
        Compute focus score using wavelet detail coefficients.

        Performs wavelet decomposition and sums absolute values of all
        detail coefficients. Higher values indicate more high-frequency
        content and better focus.

        Args:
            x: Input 2D image.

        Returns:
            Sum of absolute wavelet detail coefficients.
        """
        coeffs = pywt.wavedec2(x, "haar", level=self.level)  # 2-level decomposition
        _, (cH1, cV1, cD1), (cH2, cV2, cD2) = coeffs  # Extract high-frequency details

        # Sum of absolute values of detail coefficients (focus measure)
        focus_score = (
            np.sum(np.abs(cH1)) + np.sum(np.abs(cV1)) + np.sum(np.abs(cD1)) + np.sum(np.abs(cH2)) + np.sum(np.abs(cV2)) + np.sum(np.abs(cD2))
        )
        return focus_score


class BestFocusSelector(EDoF):
    """
    Selects the best-focused z-plane based on focus metrics.

    Unlike blending methods, this selector chooses discrete z-planes per
    pixel or region. Preserves fine details but may introduce discontinuities
    at focus boundaries.

    Args:
        single_cycle_channel: Use single reference channel for focus detection.
        crop: Pixels to crop from image borders.
        focus_cycle: Reference cycle index (1-based) for focus detection.
        focus_channel: Reference channel index (1-based) for focus detection.
        focus_metric: Focus quality metric to use.
        device: GPU device ID (optional).
    """

    def __init__(
        self,
        single_cycle_channel: bool,
        crop: int,
        focus_cycle: int,
        focus_channel: int,
        focus_metric: FocusMetric,
        device: Optional[int] = None,
    ):
        super().__init__(False, device=device)
        self.single_cycle_channel = single_cycle_channel
        self.focus_cycle = focus_cycle
        self.focus_channel = focus_channel
        self.metric = focus_metric
        self.crop = crop

    def setup_dataset(self, ds: CodexDataset) -> CodexDataset:
        ds.group_zslices()
        ds.group_channels()
        ds.group_cycles()
        return ds

    def preprocess_image(self, img, meta=None):
        return img.reshape(meta.n_cycles, meta.n_channels, meta.n_zplanes, *img.shape[1:])

    def save(self, out_dir, img, region, cycle, tile, channel):
        save_raw_tile(out_dir, img, tile, region)

    def __call__(self, tile: np.ndarray, **kwargs) -> np.ndarray:
        """
        Select best focal planes from a multi-dimensional tile.

        Evaluates focus quality across z-planes and selects the best-focused
        plane based on the configured metric. Supports single reference channel
        mode or independent per-channel evaluation.

        Args:
            tile: 5D array with shape (cycles, channels, z, height, width).
            **kwargs: Unused.

        Returns:
            Tile with best focal planes, shape (cycles, channels, 1, height, width).
            Cropped if crop parameter is specified.
        """
        if self.single_cycle_channel:
            img = tile[self.focus_cycle - 1, self.focus_channel - 1]
            nz = img.shape[0]

            scores = np.array([self.metric(img[z]) for z in range(nz)])
            best_z = np.argmax(scores)
            best_focus_tile = tile[:, :, [best_z]]
            log.debug("Best focal plane: z = {} (score: {})".format(best_z, scores.max()))
        else:
            n_cycles, n_channels, nz, h, w = tile.shape
            best_focus_tile = np.empty((n_cycles, n_channels, 1, h, w), dtype=tile.dtype)

            for cyc in range(n_cycles):
                for ch in range(n_channels):
                    img = tile[cyc, ch]
                    scores = np.array([self.metric(img[z]) for z in range(nz)])
                    best_z = np.argmax(scores)
                    best_focus_tile[cyc, ch, 0] = img[best_z]
                    log.debug("Best focal plane: z = {} (score: {})".format(best_z, scores.max()))

        if self.crop > 0:
            return best_focus_tile[:, :, :, self.crop : -self.crop, self.crop : -self.crop]
        else:
            return best_focus_tile
