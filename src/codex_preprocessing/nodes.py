import abc
import logging
import os
import os.path as osp
import time
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from codex_preprocessing._constants import CodexFiles
from codex_preprocessing.data import CodexDataset
from codex_preprocessing.io import save_raw_img, save_raw_tile_stack, save_raw_zstack
from codex_preprocessing.modules.background_correction import BackgroundCorrector
from codex_preprocessing.modules.deconvolution import Deconvolution
from codex_preprocessing.modules.edof import EDoF
from codex_preprocessing.modules.illumination import IlluminationCorrector
from codex_preprocessing.modules.stitching import Stitching
from codex_preprocessing.modules.tma_dearray import CoreographDearray
from codex_preprocessing.utils import ensure_path

log = logging.getLogger(__name__)


class Node(abc.ABC):
    """
    Abstract base class for CODEX preprocessing pipeline nodes.

    Args:
        out_dir (str or Path): Base output directory for all preprocessing results.
        subdir (str): Subdirectory under `out_dir` for this node's outputs.
        ds (CodexDataset): Dataset to process. Must have exactly one region.
        algorithm (Any): Algorithm instance or factory for processing.
        devices (Sequence[int], optional): Device IDs for parallel execution.
        n_jobs (int, optional): Number of parallel jobs if `devices` is None.
        rank (int): Worker rank, affects verbosity.

    Raises:
        AssertionError: If dataset does not have exactly one unique region.
        ValueError: If neither `devices` nor `n_jobs` is specified.
    """

    def __init__(
        self,
        out_dir,
        subdir: str,
        ds: CodexDataset,
        algorithm: Any,
        devices: Optional[int] = None,
        n_jobs: Optional[int] = None,
        rank: int = 0,
    ):
        super().__init__()
        assert len(ds.get_unique_regions()) == 1, "Dataset must contain exactly one region."
        self.ds = ds
        self.is_lazy = self.ds.is_lazy()
        self.meta = self.ds.meta
        self.rank = rank
        self.algorithm = algorithm
        self.task_name = subdir

        if devices is not None:
            self.devices = devices
            self.n_jobs = len(devices)
        elif n_jobs is not None:
            self.n_jobs = n_jobs
            self.devices = list(range(n_jobs))
        else:
            raise ValueError("Must specify either devices or n_jobs.")

        self.out_dir = osp.join(out_dir, subdir)
        os.makedirs(self.out_dir, exist_ok=True)

    def set_lazy(self, lazy: bool):
        self.is_lazy = lazy
        self.ds.set_lazy(lazy)

    def prepare_algorithm(self, algorithm: Any, device: int) -> Any:
        return algorithm

    def process(self, indices: np.ndarray, algorithm: Any, verbose: bool, device: int):
        algorithm = self.prepare_algorithm(algorithm, device)

        for idx in tqdm(indices, disable=not verbose, desc=self.task_name):
            self.process_batch(self.ds[idx], idx, algorithm)

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: Any):
        pass

    def finalize(self):
        self.ds.meta.save(self.out_dir)

    def run(self):
        # Adjust n_jobs if it exceeds dataset size
        n_splits = self.n_jobs
        if self.n_jobs > len(self.ds):
            log.warning(f"Number of jobs ({self.n_jobs}) exceeds dataset size ({len(self.ds)}). " f"Using {len(self.ds)} jobs instead.")
            n_splits = len(self.ds)

        if n_splits > 1:
            log.info(f"Running {self.task_name} in parallel on {n_splits} jobs.")
            splits = np.array_split(range(len(self.ds)), n_splits)

            Parallel(n_jobs=n_splits)(
                delayed(self.process)(
                    indices,
                    deepcopy(self.algorithm),
                    verbose=(k == self.rank),
                    device=self.devices[k % len(self.devices)],
                )
                for k, indices in enumerate(splits)
            )
        else:
            log.info(f"Running {self.task_name} sequentially.")
            self.process(
                np.arange(len(self.ds)),
                self.algorithm,
                verbose=True,
                device=self.devices[0],
            )

        self.finalize()


class DeconvolutionNode(Node):
    """
    Deconvolution processing node for CODEX images.

    This node performs deconvolution on CODEX image stacks to improve spatial
    resolution and reduce blur artifacts. It uses point spread function (PSF)
    models based on imaging parameters and supports parallel processing across
    multiple GPU devices.

    Args:
        algorithm (Callable): Factory for Deconvolution instance.
        psf (dict): PSF configuration (keys: 'size_y', 'size_x').
        skip_channels (Sequence[str]): Channels to skip deconvolution.
        ds (CodexDataset): Dataset containing z-stack images.
        out_dir (str or Path): Output directory.
        devices (Sequence[int]): Device IDs for parallel processing.
    """

    def __init__(
        self,
        algorithm: Callable,
        psf: Dict[str, int],
        skip_channels: Sequence[str],
        ds: CodexDataset,
        out_dir: str,
        devices: Sequence[int],
    ):
        super().__init__(out_dir, CodexFiles.DECONVOLUTION, ds, algorithm, devices)

        self.psf = psf
        self.skip_channels = set(skip_channels)

        self.ds.group_zslices()

        self.psf_args = {
            "size_z": self.meta.n_zplanes,
            "size_y": psf["size_y"],
            "size_x": psf["size_x"],
            "mag": self.meta.magnification,
            "na": self.meta.aperture,
            "res_axial_um": self.meta.pitch_um,
            "res_lateral_nm": self.meta.resolution_nm,
            "immersion": self.meta.immersion,
            "wavelength_nm": self.meta.wavelength_nm,
        }

    def prepare_algorithm(self, algorithm: Callable, device: int) -> Deconvolution:
        algo: Deconvolution = algorithm(device=device)
        algo.generate_psfs(**self.psf_args)
        return algo

    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: Deconvolution):
        img = batch["img"].compute() if self.is_lazy else batch["img"]
        channel = batch["channel"]
        cycle = batch["cycle"]

        if self.ds.meta.marker_name(cycle, channel) not in self.skip_channels:
            img = algorithm(img, channel)
        save_raw_zstack(self.out_dir, img, batch["region"], cycle, batch["tile"], channel, batch["zslice"])


class EDoFNode(Node):
    """
    Extended Depth of Field processing node for CODEX images.

    Args:
        algorithm (EDoF): EDoF algorithm instance.
        ds (CodexDataset): Dataset with z-stack images.
        out_dir (str or Path): Output directory.
        devices (Sequence[int]): Device IDs for parallel processing.
    """

    def __init__(self, algorithm: EDoF, ds: CodexDataset, out_dir: str, devices: Sequence[int]):
        super().__init__(out_dir, CodexFiles.EDOF, ds, algorithm, devices)
        self.ds = self.algorithm.setup_dataset(self.ds)

    def prepare_algorithm(self, algorithm: EDoF, device: int) -> EDoF:
        algorithm.set_device(device)
        return algorithm

    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: EDoF):
        img = batch["img"].compute() if self.is_lazy else batch["img"]

        img = algorithm.preprocess_image(img, self.meta)
        img = algorithm(img)
        algorithm.save(self.out_dir, img, batch["region"], batch["cycle"], batch["tile"], batch["channel"])

    def finalize(self):
        self.ds.meta.set_zplanes(1)
        self.ds.meta.save(self.out_dir)


class IlluminationCorrectionNode(Node):
    """
    Illumination correction processing node for CODEX images.

    Args:
        algorithm (IlluminationCorrector): Correction algorithm.
        ds (CodexDataset): Dataset with images to correct.
        n_jobs (int): Number of parallel jobs.
        out_dir (str or Path): Output directory.
    """

    def __init__(
        self,
        algorithm: IlluminationCorrector,
        ds: CodexDataset,
        n_jobs: int,
        out_dir: str,
    ):
        super().__init__(out_dir, CodexFiles.ILLUMINATION_CORRECTION, ds, algorithm, n_jobs=n_jobs)
        self.ds.group_tiles()

    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: IlluminationCorrector):
        img = batch["img"].compute() if self.is_lazy else batch["img"]
        cycle = batch["cycle"]
        channel = batch["channel"]

        res = algorithm(img)
        algorithm.save_field_imgs(ensure_path(self.out_dir) / "debug", cycle, channel)
        save_raw_tile_stack(self.out_dir, res, batch["region"], cycle, channel, zslice=1)


class StitchingNode(Node):
    """
    Image stitching processing node for CODEX tiles.

    Args:
        algorithm: Stitching algorithm class (not instance).
        out_fmt (Literal["raw", "codex_proc"]): Output format specification.
        extra_args: Additional arguments passed to the stitching algorithm.
        ds: CODEX dataset containing tiles to stitch.
        out_dir (str): Output directory for stitched images.
    """

    def __init__(
        self,
        algorithm: Callable,
        ds: CodexDataset,
        out_dir: str,
        n_jobs: int,
        out_fmt: Literal["raw", "codex_proc"] = "raw",
        extra_args: Optional[str] = None,
    ):
        super().__init__(out_dir, CodexFiles.STITCHING, ds, algorithm, n_jobs=n_jobs)

        self.out_fmt = out_fmt
        self.extra_args = extra_args

        self.algorithm = algorithm(out_dir=self.out_dir, ds=self.ds)
        self.ds.group_tiles()

    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: Stitching):
        img = batch["img"].compute() if self.is_lazy else batch["img"]
        res = algorithm(img)
        save_raw_img(self.out_dir, res, batch["region"], batch["cycle"], batch["channel"], tile=1, zslice=batch["zslice"])

    def process(self, indices: np.ndarray, algorithm: Any, verbose: bool, device: int):
        algorithm: Stitching = self.prepare_algorithm(algorithm, device)

        if algorithm.is_cmd:
            algorithm.set_extra_args(self.extra_args)
            algorithm()
        else:
            for idx in tqdm(indices, disable=not verbose, desc=self.task_name):
                self.process_batch(self.ds[idx], idx, algorithm)

    def finalize(self):
        self.ds.meta.save(self.out_dir)
        self.reformat()

    def reformat(self):
        """
        Convert stitched images to the specified output format.

        Raises:
            ValueError: If output format is not supported.
        """
        if self.algorithm.is_cmd:
            if self.out_fmt == "raw":
                self.algorithm.reformat_raw_fmt()
            elif self.out_fmt == "codex_proc":
                self.algorithm.reformat_codex_proc()
            else:
                raise ValueError(f"Unknown format: {self.out_fmt}")


class BackgroundCorrectionNode(Node):
    """
    Background subtraction processing node for CODEX images.

    Args:
        algorithm: Background subtraction algorithm class (not instance).
        ds: CODEX dataset containing images for background subtraction.
        out_dir (str): Output directory for background-corrected images.
    """

    def __init__(self, algorithm: Callable, ds: CodexDataset, out_dir: str, n_jobs: int):
        super().__init__(out_dir, CodexFiles.BACKGROUND_CORRECTION, ds, algorithm, n_jobs=n_jobs)
        self.set_lazy(True)

        self.algorithm: BackgroundCorrector = algorithm(ds=ds)
        self.ds = self.algorithm.setup_dataset(self.ds)

    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: BackgroundCorrector):
        img = batch["img"]
        cycle = batch["cycle"]
        channel = batch["channel"]

        out = algorithm(img, cycle, channel)
        algorithm.save(self.out_dir, out, batch["region"], batch["cycle"], batch["channel"], batch["tile"], batch["zslice"])


# class OmeTifExportNode(Node):
#     """
#     OME-TIFF export node for CODEX images.

#     Converts processed CODEX images to OME-TIFF format for compatibility
#     with standard image analysis tools and long-term archival.

#     Args:
#         algorithm: OME-TIFF conversion algorithm instance or factory.
#         ds: CODEX dataset containing processed images to export.
#         out_dir: Output directory for OME-TIFF files.
#         n_jobs: Number of parallel jobs for processing.
#     """

#     def __init__(self, algorithm: SopaExporter, ds: CodexDataset, out_dir: str):
#         super().__init__(out_dir, CodexFiles.OMETIF, ds, algorithm, n_jobs=1)

#         self.algorithm = algorithm(out_dir=self.out_dir)
#         self.ds = self.algorithm.setup_dataset(self.ds)
#         self.set_lazy(True)

#     def process(self, indices: np.ndarray, algorithm: Any, verbose: bool, device: int):
#         algorithm: SopaExporter = self.prepare_algorithm(algorithm, device)
#         algorithm(self.ds)

#     def process_batch(self, batch, batch_idx, algorithm):
#         pass


class TMADearrayNode(Node):
    def __init__(self, algorithm: CoreographDearray, ds: CodexDataset, out_dir: str, n_jobs: int, ref_cycle: int, ref_channel: int):
        super().__init__(out_dir, CodexFiles.TMA_DEARRAY, ds, algorithm, n_jobs=n_jobs)
        self.set_lazy(True)

        self.ds.sort_by(["cycle", "channel"])
        ds_ref = ds.make_copy({"cycle": ref_cycle, "channel": ref_channel})
        assert len(ds_ref) == 1, "Reference dataset must contain exactly one image."

        self.ref_results = self.algorithm.detect_cores_from_reference(ds_ref[0]["img"].squeeze().compute(), self.out_dir)

    def process_batch(self, batch: Dict[str, Any], batch_idx: int, algorithm: CoreographDearray):
        img = batch["img"].squeeze()

        algorithm(
            img,
            self.ref_results,
            cycle=batch["cycle"],
            channel=batch["channel"],
            tile=batch["tile"],
            zslice=batch["zslice"],
            out_dir=self.out_dir,
        )
