# CODEX Preprocessing

A modular, GPU-accelerated image processing pipeline for multiplexed immunofluorescence imaging data from [CODEX](https://www.akoyabio.com/codex/) (Co-Detection by indexing) systems. It transforms raw multi-cycle, multi-channel microscopy images into analysis-ready [SpatialData](https://spatialdata.scverse.org/) datasets through a configurable sequence of processing steps.

## Key Features

- **Richardson-Lucy Deconvolution** — restores image sharpness using Gibson-Lanni PSF models with GPU support via [flowdec](https://github.com/hammerlab/flowdec)
- **Extended Depth of Field (EDoF)** — collapses z-stacks into single focused images using Sobel or Dual-Tree Complex Wavelet methods
- **Illumination Correction** — removes spatial shading artifacts with the [BaSiC](https://doi.org/10.1038/ncomms14836) algorithm
- **Tile Stitching** — assembles overlapping tiles into seamless mosaics via [Ashlar](https://github.com/labsyspharm/ashlar) or a built-in M2Stitch module
- **Background Correction** — subtracts autofluorescence using linear interpolation or adaptive probe-based modeling
- **TMA Dearraying** — automatically detects and extracts tissue cores from Tissue Microarrays using a U-Net segmentation model ([Coreograph](https://github.com/HMS-IDAC/UNetCoreograph))
- **SpatialData Export** — writes processed images to the [SpatialData](https://spatialdata.scverse.org/) Zarr format with multi-resolution pyramids

All steps are optional and independently configurable through [Hydra](https://hydra.cc/).

## Requirements

- Python ≥ 3.11
- CUDA 12-capable GPU (recommended for deconvolution and EDoF)
- ~100 GB RAM for large TMA datasets

## Installation

### 1. Create the conda environment

```bash
conda env create -f env_cuda12.yml
conda activate codex_prep
```

### 2. Install the package

```bash
pip install -e .
```

### 3. Install additional GPU dependencies

PyTorch with CUDA 12.4 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

[pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) (required for wavelet-based EDoF):

```bash
pip install git+https://github.com/fbcotter/pytorch_wavelets
```

## Usage

### Quick Start

Run the full pipeline on a CODEX dataset:

```bash
python main.py data.root_dir=/path/to/raw/data
```

### Using Experiment Configs

Predefined experiment configurations live in `config/experiment/`. Run one with the `+experiment` flag:

```bash
python main.py +experiment=preprocess_ccc
```

### Skipping Pipeline Steps

Disable individual steps using Hydra's `~` (delete) syntax:

```bash
python main.py +experiment=preprocess_ccc \
    ~pipeline.deconvolution \
    ~pipeline.stitching
```

### Overriding Parameters

Override any config value from the command line:

```bash
python main.py +experiment=preprocess_ccc \
    pipeline.deconvolution.algorithm.n_iter=30 \
    pipeline.deconvolution.algorithm.use_gpu=true \
    pipeline.remove_intermediate=true
```

## Configuration

The pipeline is configured via [Hydra](https://hydra.cc/) YAML files under `config/`.

```
config/
├── preprocess.yaml              # Main config (entry point)
├── data/
│   └── raw_codex.yaml           # Dataset configuration
├── experiment/
│   └── preprocess_ccc.yaml      # Experiment presets
├── pipeline/
│   ├── default.yaml             # Pipeline defaults
│   ├── deconvolution/           # Deconvolution settings
│   ├── edof/                    # EDoF algorithm selection
│   ├── illumination_correction/ # BaSiC parameters
│   ├── stitching/               # Ashlar / M2Stitch settings
│   ├── background_correction/   # Background subtraction
│   ├── tma_dearray/             # Core detection parameters
│   └── data_export/             # SpatialData export options
└── hydra/
    └── default.yaml             # Hydra runtime settings
```

### Data Configuration

Point the pipeline to your raw CODEX data directory. The expected layout follows the standard CODEX file convention:

```
raw_data/
├── experimentV4.json                # Experiment metadata
├── cyc001_reg001/
│   ├── 1_00001_Z001_CH1.tif
│   ├── 1_00001_Z001_CH2.tif
│   └── ...
├── cyc002_reg001/
│   └── ...
└── ...
```

Set the root directory in your config or on the command line:

```yaml
# config/data/raw_codex.yaml
_target_: codex_preprocessing.data.CodexDataset
root_dir: ???             # Required — path to raw data
mode: raw                 # "raw" or "proc" (CODEX Processor format)
lazy_loading: false       # Enable dask-based lazy loading for large datasets
read_markers: false       # Load marker names from metadata
```

### Pipeline Configuration

Toggle steps and select algorithms in `config/pipeline/default.yaml`:

```yaml
defaults:
  - deconvolution: default
  - edof: focus_whiten       # or: focus_wavelet
  - illumination_correction: default
  - stitching: default
  - background_correction: default
  - tma_dearray: null        # Enable with: tma_dearray: default
  - data_export: default

remove_intermediate: false   # Delete intermediate outputs to save disk space
```

## Project Structure

```
├── main.py                         # Entry point
├── config/                         # Hydra configuration files
├── src/codex_preprocessing/
│   ├── pipeline.py                 # Pipeline orchestration
│   ├── nodes.py                    # Abstract node / parallel execution logic
│   ├── data/
│   │   ├── dataset.py              # CodexDataset class
│   │   └── metadata.py             # Experiment metadata parser
│   ├── io/
│   │   ├── reader.py               # Raw & processed data readers
│   │   └── writer.py               # TIFF writers
│   ├── modules/                    # Processing algorithms
│   │   ├── deconvolution.py        # Richardson-Lucy deconvolution
│   │   ├── edof.py                 # Extended depth of field
│   │   ├── illumination.py         # BaSiC illumination correction
│   │   ├── stitching.py            # Ashlar / M2Stitch stitching
│   │   ├── background_correction.py
│   │   ├── tma_dearray.py          # Coreograph TMA dearraying
│   │   └── spatialdata_exporter.py # SpatialData Zarr export
│   ├── models/coreograph/          # U-Net model for TMA segmentation
│   └── utils/                      # Image & general utilities
├── weights/coreograph/             # Pre-trained U-Net weights
├── notebooks/                      # Jupyter notebooks for testing individual steps
├── env_cuda12.yml                  # Conda environment (CUDA 12)
└── pyproject.toml                  # Package metadata
```

## Notebooks

Interactive Jupyter notebooks are provided in `notebooks/` for testing and debugging individual pipeline steps:

| Notebook | Purpose |
|----------|---------|
| `test_deconvolution.ipynb` | Richardson-Lucy deconvolution |
| `test_edof.ipynb` | Extended depth of field |
| `test_illumination.ipynb` | BaSiC illumination correction |
| `test_stitching.ipynb` | Tile stitching |
| `test_bg_sub.ipynb` | Background subtraction |
| `test_dearray.ipynb` | TMA dearraying |
| `test_ometif.ipynb` | OME-TIFF export |

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and ensure existing functionality is preserved
4. Submit a pull request
