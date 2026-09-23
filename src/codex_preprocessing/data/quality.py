from typing import List, Optional

import numpy as np
import pandas as pd

from codex_preprocessing.data.metadata import Metadata

# Refractive index per immersion medium, matching Deconvolution.get_immersion_ri
_IMMERSION_RI = {"air": 1.0, "water": 1.33, "oil": 1.5115}


def check_completeness(df: pd.DataFrame, meta: Metadata) -> List[str]:
    """
    Check that every (region, cycle, tile, channel, zslice) combination expected
    from experimentV4.json has a matching file in the raw dataset.

    Args:
        df (pd.DataFrame): DataFrame produced by read_raw_data.
        meta (Metadata): Parsed experimentV4.json metadata.

    Returns:
        List[str]: One message per missing combination.
    """
    found = set(zip(df["region"], df["cycle"], df["tile"], df["channel"], df["zslice"]))

    issues = []
    for region in meta.regions:
        for cycle in range(1, meta.n_cycles + 1):
            for tile in range(1, meta.n_tiles(region) + 1):
                for channel in range(1, meta.n_channels + 1):
                    for zslice in range(1, meta.n_zplanes + 1):
                        if (region, cycle, tile, channel, zslice) not in found:
                            issues.append(f"Missing file: region={region} cycle={cycle} tile={tile} channel={channel} zslice={zslice}")
    return issues


def check_metadata_consistency(df: pd.DataFrame, meta: Metadata) -> List[str]:
    """
    Check that the regions/cycles/channels found on disk match what
    experimentV4.json declares.

    Args:
        df (pd.DataFrame): DataFrame produced by read_raw_data or read_processed_data.
        meta (Metadata): Parsed experimentV4.json metadata.

    Returns:
        List[str]: One message per mismatched count.
    """
    issues = []

    n_regions_found = df["region"].nunique()
    if n_regions_found != meta.n_regions:
        issues.append(f"Found {n_regions_found} region(s) on disk, but metadata declares {meta.n_regions}.")

    n_cycles_found = df["cycle"].nunique()
    if n_cycles_found != meta.n_cycles:
        issues.append(f"Found {n_cycles_found} cycle(s) on disk, but metadata declares {meta.n_cycles}.")

    n_channels_found = df["channel"].nunique()
    if n_channels_found != meta.n_channels:
        issues.append(f"Found {n_channels_found} channel(s) on disk, but metadata declares {meta.n_channels}.")

    return issues


def check_nyquist_sampling(meta: Metadata) -> List[str]:
    """
    Check whether the acquired pixel/voxel size satisfies the Nyquist sampling
    criterion for the given optics, using the same widefield formulas as
    Huygens' "Test Sampling Density" QC step (Pawley, Handbook of Biological
    Confocal Microscopy, 3rd ed.):

        lateral Nyquist distance = wavelength / (4 * NA)
        axial Nyquist distance   = wavelength * n / NA**2

    A dataset is undersampled (unsuitable for reliable deconvolution) when the
    acquired pixel/voxel size is larger than the corresponding Nyquist distance.

    Args:
        meta (Metadata): Parsed experimentV4.json metadata.

    Returns:
        List[str]: One message per undersampled channel/axis.

    Raises:
        KeyError: If meta.immersion is not one of "air", "water", "oil".
    """
    na = meta.aperture
    n = _IMMERSION_RI[meta.immersion]
    px_lateral_um = meta.resolution_nm / 1000.0
    px_axial_um = meta.pitch_um

    issues = []
    for wavelength_nm in meta.wavelength_nm:
        wavelength_um = wavelength_nm / 1000.0
        nyquist_lateral_um = wavelength_um / (4 * na)
        nyquist_axial_um = wavelength_um * n / na**2

        if px_lateral_um > nyquist_lateral_um:
            issues.append(
                f"Lateral pixel size {px_lateral_um:.3f} um exceeds the Nyquist limit "
                f"{nyquist_lateral_um:.3f} um for wavelength {wavelength_nm:.0f} nm (undersampled)."
            )
        if px_axial_um > nyquist_axial_um:
            issues.append(
                f"Axial step {px_axial_um:.3f} um exceeds the Nyquist limit "
                f"{nyquist_axial_um:.3f} um for wavelength {wavelength_nm:.0f} nm (undersampled)."
            )
    return issues


def check_saturation(img: np.ndarray, meta: Metadata, threshold: float = 0.001) -> Optional[str]:
    """
    Check an image for clipped/saturated pixels, mirroring the saturation
    warning Huygens raises before deconvolution.

    Args:
        img (np.ndarray): Image array to check.
        meta (Metadata): Parsed experimentV4.json metadata (used for the bit-depth max value).
        threshold (float): Maximum tolerated fraction of saturated pixels before flagging.

    Returns:
        Optional[str]: Warning message if the saturated fraction exceeds threshold, else None.
    """
    max_val = np.iinfo(meta.dtype).max
    frac_saturated = float(np.mean(img >= max_val))
    if frac_saturated > threshold:
        return f"{frac_saturated:.2%} of pixels are saturated (>= {max_val})."
    return None
