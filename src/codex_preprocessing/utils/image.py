from typing import Union

import numpy as np


def clip_and_cast_to_uint(img: np.ndarray, dtype: Union[type, np.dtype, str]) -> np.ndarray:
    """
    Clip and cast an image array to an unsigned integer data type.

    This function converts an image array to the specified unsigned integer type
    by clipping values to the valid range of that type and then casting. No
    intensity scaling or normalization is performed - values outside the valid
    range are simply clipped.

    Args:
        img: input image array of any numeric type.
        dtype: target unsigned integer data type. Must be one of: ``np.uint8``, ``np.uint16``,
            ``np.uint32``, or ``np.uint64``. Can be specified as a type, numpy dtype, or string.
            Defaults to ``np.uint8``.

    Returns:
        Array with values clipped to the valid range of ``dtype`` and cast to that type.
    """
    target_dtype = np.dtype(dtype)
    if not np.issubdtype(target_dtype, np.unsignedinteger):
        raise ValueError(f"Only unsigned integer types are valid for conversion. " f"Got: {target_dtype}")

    # Get valid range for target dtype
    dtype_info = np.iinfo(target_dtype)
    min_val, max_val = dtype_info.min, dtype_info.max

    # Clip values to valid range and cast to target type
    return np.clip(img, min_val, max_val).astype(target_dtype)
