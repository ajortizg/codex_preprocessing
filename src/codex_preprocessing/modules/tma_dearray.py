import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from skimage import color, exposure, feature, filters, measure, morphology, segmentation

from codex_preprocessing.io import save_raw_img
from codex_preprocessing.models.coreograph import imtools
from codex_preprocessing.models.coreograph.unet import UNet2D
from codex_preprocessing.utils import ensure_path

log = logging.getLogger(__name__)


class TMADearray:
    def __init__(self):
        pass


class CoreographDearray(TMADearray):

    def __init__(
        self,
        model_weights: str | Path,
        down_factor: int = 5,
        # Preprocessing parameters
        preprocessing_params: dict = None,
        # Tissue mask parameters
        mask_params: dict = None,
        # Core detection parameters
        detection_params: dict = None,
        extraction_params: dict = None,
    ):
        """
        Initialize Coreograph-based TMA dearrayer.

        Args:
            model_weights: Path to trained U-Net model weights
            down_factor: Downsampling factor (2^down_factor reduction)
            preprocessing_params: Dict with keys:
                - sigma: Gaussian blur std dev (default: 0)
                - clip_limit: CLAHE clip limit (default: 0)
                - kernel_size: CLAHE kernel size (default: 8)
                - out_range: Output intensity range (default: (0, 0.983))
            mask_params: Dict with keys:
                - blur_sigma: Gaussian blur for mask (default: 5)
                - small_obj_threshold: Min object size fraction (default: 0.2)
                - buffer_factor: Diameter estimation buffer (default: 1.2)
            detection_params: Dict with keys:
                - sensitivity: Blob detection threshold (default: 0.2)
                - min_sigma_factor: Min blob sigma factor (default: 0.6)
            extraction_params: Dict with keys:
                - translate: Dict mapping label -> (y, x) translation tuple in pixels
                  Example: {1: (30, 50), 3: (-15, -20)} translates label 1 by (30, 50)
                  and label 3 by (-15, -20). Default: None (no translation)
                - translate_in_down_space: If True, translation is in downsampled
                  coordinates; if False, in original image coordinates (default: False)
        """
        super().__init__()
        self.model_weights = model_weights

        # Core parameters
        self.down_factor = down_factor
        self.scale = 1.0 / (2**down_factor)

        # Preprocessing parameters
        preproc = preprocessing_params or {}
        self.sigma = preproc.get("sigma", 0)
        self.clip_limit = preproc.get("clip_limit", 0)
        self.kernel_size = preproc.get("kernel_size", 8)
        self.out_range = tuple(preproc.get("out_range", (0, 0.983)))

        # Mask computation parameters
        mask = mask_params or {}
        self.mask_blur_sigma = mask.get("blur_sigma", 5)
        self.small_obj_threshold = mask.get("small_obj_threshold", 0.2)
        self.buffer_factor = mask.get("buffer_factor", 1.2)

        # Core detection parameters
        detect = detection_params or {}
        self.sensitivity = detect.get("sensitivity", 0.2)
        self.min_sigma_factor = detect.get("min_sigma_factor", 0.6)

        # Core extraction parameters
        extract = extraction_params or {}
        self.translate = extract.get("translate", None)
        self.translate_in_down_space = extract.get("translate_in_down_space", False)

    def prepare_image_for_inference(self, img: np.ndarray) -> np.ndarray:
        """
        Prepare TMA image for deep learning inference.

        Applies downsampling, optional smoothing, contrast enhancement, and
        intensity normalization to prepare the image for the U-Net model.

        Args:
            img: Input 2D grayscale image

        Returns:
            Preprocessed image ready for model inference
        """
        if img.ndim != 2:
            raise ValueError("Input image must be 2D array")

        # Downsample
        for _ in range(self.down_factor):
            h, w = img.shape
            new_h, new_w = int(h * 0.5), int(w * 0.5)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Optional Gaussian smoothing
        if self.sigma > 0:
            img = ndi.gaussian_filter(img, sigma=self.sigma)

        # Optional CLAHE contrast enhancement
        if self.clip_limit > 0:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.kernel_size, self.kernel_size))
            img = clahe.apply(img)

        # Normalize to [0, 1] and apply output range
        img = imtools.im2double(img)
        img = exposure.rescale_intensity(img, in_range=(np.min(img), np.max(img)), out_range=self.out_range)
        img = imtools.im2double(img)

        return img

    def predict_tissue_probability(self, img: np.ndarray) -> np.ndarray:
        """
        Predict tissue vs background probability using U-Net model.

        Args:
            img: Preprocessed 2D image from prepare_image_for_inference()

        Returns:
            Probability map where values close to 1 indicate tissue
        """
        if img.ndim != 2:
            raise ValueError("Input image must be 2D array")

        UNet2D.singleImageInferenceSetup(str(self.model_weights), -1)
        class_probs = UNet2D.singleImageInference(img, "accumulate", 1)
        UNet2D.singleImageInferenceCleanup()
        return class_probs

    def compute_tissue_mask(self, class_probs: np.ndarray) -> Dict[str, Any]:
        """
        Convert probability map to binary tissue mask with properties.

        Applies Gaussian smoothing, Otsu thresholding, morphological operations,
        and size-based filtering to create clean tissue masks. Computes core
        statistics for downstream detection.

        Args:
            class_probs: Tissue probability map from predict_tissue_probability()

        Returns:
            Dictionary containing:
                - mask: Binary tissue mask
                - core_radius: Estimated core radius (pixels, downsampled)
                - core_diameter: Estimated core diameter (pixels, downsampled)
                - median_area: Median core area
                - max_area: 99th percentile core area
        """
        # Blur and threshold
        mask = filters.gaussian(np.uint8(class_probs * 255), self.mask_blur_sigma)
        threshold = filters.threshold_otsu(mask)
        mask = mask > threshold

        # Morphological operations
        kernel_size = int(self.mask_blur_sigma * 2)
        kernel = np.ones((kernel_size, kernel_size))
        mask = morphology.binary_closing(mask, kernel)
        mask = ndi.binary_fill_holes(mask)

        # Compute region statistics
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask, cache=False)
        areas = [region.area for region in regions]

        if len(regions) < 3:
            med_area = np.median(areas)
            max_area = np.percentile(areas, 99)
        else:
            # Relabel to ensure continuous labels
            label_mask = np.zeros(mask.shape, dtype=np.uint32)
            for idx, props in enumerate(regions, start=1):
                yi, xi = props.coords[:, 0], props.coords[:, 1]
                label_mask[yi, xi] = idx

            regions = measure.regionprops(label_mask)
            areas = [region.area for region in regions]
            med_area = np.median(areas)
            max_area = np.percentile(areas, 99)

        # Remove small objects
        min_size = int(self.small_obj_threshold * med_area)
        mask = morphology.remove_small_objects(mask, min_size)

        # Calculate core statistics
        core_radius = round(np.sqrt(med_area / np.pi))
        core_diameter = round(np.sqrt(max_area / np.pi) * 2 * self.buffer_factor)

        log.info(f"Tissue mask: median_area={med_area:.0f}, " f"core_radius={core_radius}, core_diameter={core_diameter}")

        return {
            "mask": mask,
            "core_radius": core_radius,
            "core_diameter": core_diameter,
            "median_area": med_area,
            "max_area": max_area,
        }

    def segment_individual_cores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect and segment individual tissue cores using watershed.

        Uses blob detection to find core centers, then applies watershed
        segmentation to separate touching cores. Scales centroids back to
        original image coordinates.

        Args:
            mask_data: Output from compute_tissue_mask() containing:
                - mask: Binary tissue mask
                - core_radius: Estimated core radius

        Returns:
            Dictionary containing:
                - core_label: Labeled image with individual cores
                - centroids: Core centroids in original image coordinates
                - num_cores: Total number of detected cores
                - region_properties: Region properties for each core
        """
        mask = data["mask"]
        core_radius = data["core_radius"]

        # Detect potential core centers using blob detection
        min_sigma = core_radius * self.min_sigma_factor
        blobs = feature.blob_log(mask, min_sigma=min_sigma, threshold=self.sensitivity)

        # Create marker image from blob centers
        markers = np.zeros(mask.shape, dtype=np.uint8)
        for blob in blobs:
            yi = int(round(blob[0]))
            xi = int(round(blob[1]))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                markers[yi, xi] = 1

        # Apply mask and compute distance transform for watershed
        markers = markers * mask
        distances = ndi.distance_transform_edt(1 - markers)

        # Label markers and perform watershed
        labeled_markers = measure.label(markers)
        core_labels = segmentation.watershed(distances, labeled_markers, watershed_line=True, mask=mask)

        # Extract core properties
        regions = measure.regionprops(core_labels)
        log.info(f"Detected {len(regions)} individual tissue cores")

        region_dicts = [
            {
                "label": int(r.label),
                "centroid": (float(r.centroid[0]), float(r.centroid[1])),
                "area": int(r.area),
                "eccentricity": float(r.eccentricity),
                "solidity": float(r.solidity),
                "bbox": tuple(int(v) for v in r.bbox),
                "major_axis_length": float(r.major_axis_length),
                "minor_axis_length": float(r.minor_axis_length),
                "perimeter": float(r.perimeter),
            }
            for r in regions
        ]

        return {
            **data,
            "core_label": core_labels,
            "region_properties": region_dicts,
        }

    def extract_cores(self, img: da.Array, data: Dict[str, Any], compute_masks: bool) -> Dict[str, Any]:
        assert img.ndim == 2, "Input image must be 2D array"
        height, width = img.shape
        regions = data["region_properties"]
        core_label = data["core_label"]

        centroids_down, centroids_high = self._get_centroids(data, regions)
        diameter_down_x, diameter_down_y, diameter_high_x, diameter_high_y = self._get_diameter(data, regions)

        if "bboxes" in data:
            bboxes = data["bboxes"]
            compute_bboxes = False
        else:
            bboxes = {}
            compute_bboxes = True

        core_imgs = {}
        core_masks = {}
        for region in regions:
            label = region["label"]

            if compute_bboxes:
                bbox = self._compute_core_bbox(centroids_high[label], diameter_high_y[label], diameter_high_x[label], height, width)
                bboxes[label] = bbox
            else:
                bbox = bboxes[label]

            core_imgs[label] = img[bbox["y_slice"], bbox["x_slice"]]
            if compute_masks:
                core_masks[label] = self._extract_and_resize_mask(core_label, label, bbox, core_imgs[label].shape)

        res_data = {
            **data,
            "centroids_down": centroids_down,
            "centroids_high": centroids_high,
            "diameter_down_x": diameter_down_x,
            "diameter_down_y": diameter_down_y,
            "diameter_high_x": diameter_high_x,
            "diameter_high_y": diameter_high_y,
            "core_imgs": core_imgs,
            "bboxes": bboxes,
        }
        if compute_masks:
            res_data["core_masks"] = core_masks
        return res_data

    def _get_centroids(self, data, regions):
        if data.get("centroids_down", None) is None:
            centroids_down = {region["label"]: np.array(region["centroid"]) for region in regions}
            centroids_high = {region["label"]: np.array(region["centroid"]) / self.scale for region in regions}
            centroids_down, centroids_high = self._translate_centroids(centroids_down, centroids_high, regions)
        else:
            centroids_down = data["centroids_down"]
            centroids_high = data["centroids_high"]
        return centroids_down, centroids_high

    def _get_diameter(self, data, regions):
        if data.get("diameter_down_x", None) is None:
            diameter_down_x = {region["label"]: data["core_diameter"] for region in regions}
            diameter_down_y = {region["label"]: data["core_diameter"] for region in regions}
            diameter_high_x = {region["label"]: data["core_diameter"] / self.scale for region in regions}
            diameter_high_y = {region["label"]: data["core_diameter"] / self.scale for region in regions}
        else:
            diameter_down_x = data["diameter_down_x"]
            diameter_down_y = data["diameter_down_y"]
            diameter_high_x = data["diameter_high_x"]
            diameter_high_y = data["diameter_high_y"]
        return diameter_down_x, diameter_down_y, diameter_high_x, diameter_high_y

    def _compute_core_bbox(self, centroid, dy, dx, height, width) -> Dict[str, Any]:
        cy, cx = centroid

        # Compute bounding box with bounds checking
        x0 = max(round(cx - dx / 2), 0)
        x1 = min(round(cx + dx / 2), width)

        y0 = max(round(cy - dy / 2), 0)
        y1 = min(round(cy + dy / 2), height)

        return {
            "x0": int(x0),
            "x1": int(x1),
            "y0": int(y0),
            "y1": int(y1),
            "x_slice": slice(int(x0), int(x1)),
            "y_slice": slice(int(y0), int(y1)),
        }

    def _extract_and_resize_mask(self, core_label, label, bbox, target_shape):
        # Compute bounding box in downsampled space
        y0_down = int(bbox["y0"] * self.scale)
        y1_down = int(bbox["y1"] * self.scale)
        x0_down = int(bbox["x0"] * self.scale)
        x1_down = int(bbox["x1"] * self.scale)

        core = core_label == label
        mask = core[y0_down:y1_down, x0_down:x1_down]

        # Resize to match extracted image shape
        mask = mask.astype(np.uint8)
        new_h, new_w = target_shape
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    def _translate_centroids(self, centroids_down, centroids_high, regions):
        if self.translate is None:
            return centroids_down, centroids_high

        # Validate that all translation labels are valid
        all_labels = {region["label"] for region in regions}
        invalid_labels = set(self.translate.keys()) - all_labels

        if invalid_labels:
            log.warning(f"translate contains invalid labels: {invalid_labels}")

        # Apply translations to valid labels
        labels_to_translate = set(self.translate.keys()) & all_labels

        if labels_to_translate:
            space_type = "downsampled" if self.translate_in_down_space else "original"
            log.info(f"Applying centroid translations to {len(labels_to_translate)} cores ({space_type} space)")

        for label in labels_to_translate:
            ty, tx = self.translate[label]

            if ty != 0 or tx != 0:
                if self.translate_in_down_space:
                    # Translation in downsampled space
                    centroids_down[label][0] += ty
                    centroids_down[label][1] += tx
                    centroids_high[label][0] += ty / self.scale
                    centroids_high[label][1] += tx / self.scale
                else:
                    # Translation in original image space
                    centroids_down[label][0] += ty * self.scale
                    centroids_down[label][1] += tx * self.scale
                    centroids_high[label][0] += ty
                    centroids_high[label][1] += tx

                log.info(f"Label {label}: translated by (y={ty}, x={tx}) in {space_type} space")

        return centroids_down, centroids_high

    def detect_cores_from_reference(self, img: np.ndarray | da.Array, out_dir: Optional[str | Path] = None):
        if out_dir is not None:
            out_dir = ensure_path(out_dir)
            debug_dir = out_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Saving dearray debug outputs to {debug_dir}")

        log.info("Detecting cores from reference image")
        if isinstance(img, da.Array):
            img = img.compute()

        img_p = self.prepare_image_for_inference(img)
        self._save_preprocessing_results(img_p, debug_dir)

        class_probs = self.predict_tissue_probability(img_p)
        self._save_probability_results(class_probs, debug_dir)

        data = self.compute_tissue_mask(class_probs)
        self._save_mask_results(data, debug_dir)

        data = self.segment_individual_cores(data)
        self._save_segmentation_results(data, debug_dir)

        data = self.extract_cores(img, data, compute_masks=True)
        self._create_comprehensive_visualization(img_p, data, debug_dir)
        self._save_core_masks(data, out_dir)

        data["preprocessed_image"] = img_p
        self._save_detection_summary(data, img.shape, debug_dir)
        self._save_core_properties_csv(data["region_properties"], debug_dir)

        return data

    def _save_core_masks(self, data: Dict[str, Any], out_dir: Optional[Path]):
        masks_dir = out_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        core_masks = data["core_masks"]
        for lbl, mask in core_masks.items():
            tifffile.imwrite(masks_dir / f"mask_reg{lbl:03d}.tif", mask)

    def _save_preprocessing_results(self, img: np.ndarray, out_dir: Optional[Path]):
        if out_dir is None:
            return
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax = ax.flatten()
        ax[0].imshow(img, cmap="gray")
        ax[1].hist(img.ravel(), bins=100, color="steelblue", alpha=0.7)
        ax[1].set_xlabel("Intensity")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("Histogram")
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "1_preprocessed_image.png", dpi=150)
        plt.close(fig)

    def _save_probability_results(self, class_probs: np.ndarray, out_dir: Optional[Path]):
        if out_dir is None:
            return
        plt.imsave(out_dir / "2_class_probabilities.png", class_probs, cmap="hot")

    def _save_mask_results(self, data: Dict[str, Any], out_dir: Optional[Path]):
        if out_dir is None:
            return
        plt.imsave(out_dir / "3_tissue_mask.png", data["mask"], cmap="gray")

    def _save_segmentation_results(self, data: Dict[str, Any], out_dir: Optional[Path]):
        if out_dir is None:
            return
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(data["core_label"], cmap="tab20")

        # Add labels with better styling
        for region in data["region_properties"]:
            y, x = region["centroid"]
            ax.text(
                x,
                y,
                str(region["label"]),
                color="white",
                fontsize=10,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="black", alpha=0.7, edgecolor="white", linewidth=1),
            )

        ax.set_title(f'Segmented Cores (n={len(data["region_properties"])})', fontsize=16, weight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"4_segmented_cores.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_comprehensive_visualization(self, img_prep: np.ndarray, data: Dict[str, Any], path: Optional[Path]):
        if path is None:
            return

        # Convert to RGB for visualization
        img_norm = (img_prep / np.percentile(img_prep, 99.9)).clip(0, 1)
        img_rgb = color.gray2rgb((img_norm * 255).astype(np.uint8))

        regions = data["region_properties"]
        height, width = img_rgb.shape[:2]

        for region in regions:
            label = region["label"]
            bbox = self._compute_core_bbox(
                data["centroids_down"][label], data["diameter_down_y"][label], data["diameter_down_x"][label], height, width
            )
            cv2.rectangle(img_rgb, (bbox["x0"], bbox["y0"]), (bbox["x1"], bbox["y1"]), color=(255, 0, 0), thickness=2)

            cy, cx = data["centroids_down"][label]
            cv2.putText(img_rgb, str(label), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        boundaries = segmentation.find_boundaries(data["core_label"])
        img_rgb[boundaries] = (255, 255, 255)

        # Save with title
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(img_rgb)
        ax.set_title(f"Final Detection: {len(regions)} Cores", fontsize=16)
        # ax.axis("off")
        plt.tight_layout()
        plt.savefig(path / "5_detected_cores.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_detection_summary(self, data: Dict[str, Any], original_shape: tuple, out_dir: Optional[Path]) -> None:
        if out_dir is None:
            return
        summary_file = out_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("TMA DEARRAYING SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            # Input information
            f.write("Input Image:\n")
            f.write(f"  Original shape: {original_shape}\n")
            f.write(f"  Downsampling factor: {self.down_factor} (scale: 1/{2**self.down_factor})\n")
            f.write(f"  Downsampled shape: {data['preprocessed_image'].shape}\n\n")

            # Detection results
            f.write("Detection Results:\n")
            f.write(f"  Number of cores detected: {len(data['region_properties'])}\n")
            f.write(f"  Core diameter (downsampled): {data['core_diameter']} pixels\n")
            f.write(f"  Core diameter (original): {data['core_diameter'] / self.scale:.2f} pixels\n")
            f.write(f"  Median core area: {data['median_area']:.0f} pixels²\n\n")

            # Parameters
            f.write("Pipeline Parameters:\n")
            f.write(f"  Preprocessing:\n")
            f.write(f"    Gaussian sigma: {self.sigma}\n")
            f.write(f"    CLAHE clip limit: {self.clip_limit}\n")
            f.write(f"    CLAHE kernel size: {self.kernel_size}\n")
            f.write(f"    Output range: {self.out_range}\n\n")

            f.write(f"  Mask Computation:\n")
            f.write(f"    Blur sigma: {self.mask_blur_sigma}\n")
            f.write(f"    Small object threshold: {self.small_obj_threshold}\n")
            f.write(f"    Buffer factor: {self.buffer_factor}\n\n")

            f.write(f"  Core Detection:\n")
            f.write(f"    Sensitivity: {self.sensitivity}\n")
            f.write(f"    Min sigma factor: {self.min_sigma_factor}\n\n")

            # Translation information
            if self.translate is not None:
                f.write("Translations Applied:\n")
                space = "downsampled" if self.translate_in_down_space else "original"
                for label, (ty, tx) in self.translate.items():
                    if label in data["centroids_down"]:
                        f.write(f"  Core {label}: Δy={ty:+.1f}, Δx={tx:+.1f} ({space} space)\n")
                f.write("\n")

            # Core properties table
            f.write("Core Properties:\n")
            f.write(f"{'Label':<8}{'Centroid (y,x)':<25}{'Area':<12}{'Eccentricity':<15}{'Solidity':<10}\n")
            f.write("-" * 70 + "\n")
            for region in sorted(data["region_properties"], key=lambda r: r["label"]):
                cy, cx = region["centroid"]
                f.write(
                    f"{region['label']:<8}"
                    f"{f'({cy:.1f}, {cx:.1f})':<25}"
                    f"{region['area']:<12.1f}"
                    f"{region['eccentricity']:<15.3f}"
                    f"{region['solidity']:<10.3f}\n"
                )

        log.info(f"Saved detection summary to {summary_file}")

    def _save_core_properties_csv(self, regions: list, out_dir: Optional[Path]):
        if out_dir is None:
            return

        properties_list = []
        for region in regions:
            properties_list.append(
                {
                    "label": region["label"],
                    "centroid_y": region["centroid"][0],
                    "centroid_x": region["centroid"][1],
                    "area": region["area"],
                    "perimeter": region["perimeter"],
                    "eccentricity": region["eccentricity"],
                    "solidity": region["solidity"],
                    "major_axis_length": region["major_axis_length"],
                    "minor_axis_length": region["minor_axis_length"],
                }
            )

        df = pd.DataFrame(properties_list)
        csv_file = out_dir / "core_properties.csv"
        df.to_csv(csv_file, index=False)

        log.info(f"Saved core properties to {csv_file}")

    def __call__(
        self, img: np.ndarray | da.Array, data: Dict[str, Any], cycle: int, channel: int, tile: int, zslice: int, out_dir: Optional[str | Path]
    ):
        """
        Execute complete dearraying pipeline.

        Args:
            img: Input TMA image (2D grayscale)

        Returns:
            Dictionary with all pipeline outputs including tissue masks,
            labeled cores, centroids, and region properties
        """
        data = self.extract_cores(img, data, compute_masks=False)
        lazy = isinstance(img, da.Array)

        if out_dir is None:
            return data
        else:
            out_dir: Path = ensure_path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        core_imgs = data["core_imgs"]
        for lbl, core in core_imgs.items():
            save_raw_img(out_dir, img=core.compute() if lazy else core, region=lbl, cycle=cycle, channel=channel, tile=tile, zslice=zslice)
