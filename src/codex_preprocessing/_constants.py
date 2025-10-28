class CodexFiles:
    SEG_TMP_DIR = "seg_temp.zarr"
    PREPROCESSING_DIR = "preprocessing"

    # Raw data format
    # RAW_DATA_RE = (
    #     r"cyc(?P<cycle>[0-9]{3})_reg(?P<region>[0-9]{3})/.*\w_(?P<tile>\d*)_Z(?P<zslice>\d*)_CH(?P<channel>\d).tif"
    # )
    RAW_DATA_RE = r"cyc(?P<cycle>[0-9]{3})_reg(?P<region>[0-9]{3})/.*\w_(?P<tile>\d*)_Z(?P<zslice>\d*)_CH(?P<channel>\d)\.tif(?:f\.tiff)?$"
    # PIPE_DATA_RE = r"cyc(?P<cycle>[0-9]{3})_reg(?P<region>[0-9]{3})/.*\w_(?P<tile>\d*)_Z(?P<zslice>\d*)_CH(?P<channel>\d).tiff.tiff"
    RAW_DIR_RE = r"cyc(?P<cycle>[0-9]{3})_reg(?P<region>[0-9]{3})"
    RAW_DIR_FMT = "cyc{cycle:03d}_reg{region:03d}"
    RAW_IMG_FMT = "{region:01d}_{tile:05d}_Z{zslice:03d}_CH{channel:01d}.tif"

    # Preprocessed data format
    PROC_IMG_RE = r"reg(?P<region>[0-9]{3})_X(?P<x>[0-9]{2})_Y(?P<y>[0-9]{2})_t(?P<cycle>[0-9]{3})_z(?P<zslice>[0-9]*)_c(?P<channel>\d{3}).tif"
    PROC_DIR_FMT = "reg{region:03d}_t{cycle:03d}"
    PROC_IMG_FMT = "reg{region:03d}_X{x:02d}_Y{y:02d}_t{cycle:03d}_z{z:03d}_c{channel:03d}.tif"

    # Ashlar data format
    STITCHED_TIF_FMT = "reg{region:03d}_cyc{cycle}_ch{channel}.tif"
    STITCHED_TIF_RE = r"reg(?P<region>[0-9]{3})_cyc(?P<cycle>[0-9]{3})_ch(?P<channel>[0-9]{3}).tif"
    # STITCHED_CODEX_PROC_TIF_RE = (
    #     r"reg(?P<region>[0-9]{3})_cyc(?P<cycle>[0-9]{3})_ch(?P<channel>[0-9]{3})_(?P<marker>[A-Za-z0-9\s\.\-]+)\.tif"
    # )

    STITCHED_CODEX_PROC_TIF_RE = (
        r"reg(?P<region>[0-9]{3})_cyc(?P<cycle>[0-9]{3})_ch(?P<channel>[0-9]{3})" r"(?:_(?P<marker>[A-Za-z0-9\s\.\-]+))?\.tif"
    )
    STITCHED_CODEX_PROC_TIF_FMT = "reg{region:03d}_cyc{cycle}_ch{channel}_{marker}.tif"

    STITCHED_OME_TIF = "stitched.ome.tif"

    ZARR_REGION_RE = r"reg(?P<region>[0-9]{3})(?:[a-zA-Z0-9_]*).zarr"

    # Metadata files
    # META_JSON_FILE = "experiment.json"
    META_JSON_V4_FILE = "experimentV4.json"

    # preprocessing subdirs
    DECONVOLUTION = "deconvolution"
    EDOF = "edof"
    ILLUMINATION_CORRECTION = "illumination_correction"
    STITCHING = "stitching"
    BACKGROUND_CORRECTION = "background_correction"
    TMA_DEARRAY = "tma_dearray"
