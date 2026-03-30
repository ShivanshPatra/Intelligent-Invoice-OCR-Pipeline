"""
preprocessor.py
---------------
OpenCV-based image preprocessing pipeline.
Handles deskewing, denoising, and thresholding to maximize Tesseract accuracy
on real-world noisy invoice scans.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    image: np.ndarray
    was_deskewed: bool
    skew_angle: float
    preprocessing_steps: list[str]


class ImagePreprocessor:
    """
    Prepares invoice images for Tesseract OCR.
    
    Pipeline:
        1. Convert to grayscale
        2. Deskew (correct rotation from scanning)
        3. Denoise
        4. Adaptive threshold (handles uneven lighting)
        5. Morphological cleanup
    """

    def __init__(
        self,
        deskew: bool = True,
        denoise: bool = True,
        threshold_method: str = "adaptive",  # "adaptive" | "otsu"
        morph_cleanup: bool = True,
    ):
        self.deskew = deskew
        self.denoise = denoise
        self.threshold_method = threshold_method
        self.morph_cleanup = morph_cleanup

    def process(self, image: np.ndarray) -> PreprocessingResult:
        """
        Run full preprocessing pipeline on an image.

        Args:
            image: BGR numpy array (as loaded by OpenCV)

        Returns:
            PreprocessingResult with processed image and metadata
        """
        steps = []
        skew_angle = 0.0
        was_deskewed = False

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            steps.append("grayscale")
        else:
            gray = image.copy()

        # 2. Deskew
        if self.deskew:
            gray, skew_angle = self._deskew_image(gray)
            was_deskewed = abs(skew_angle) > 0.5
            if was_deskewed:
                steps.append(f"deskew({skew_angle:.2f}deg)")

        # 3. Denoise
        if self.denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            steps.append("denoise")

        # 4. Threshold
        if self.threshold_method == "adaptive":
            processed = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            steps.append("adaptive_threshold")
        elif self.threshold_method == "otsu":
            _, processed = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            steps.append("otsu_threshold")
        else:
            processed = gray

        # 5. Morphological cleanup (remove small noise artifacts)
        if self.morph_cleanup:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            steps.append("morph_cleanup")

        logger.debug(f"Preprocessing steps applied: {steps}")

        return PreprocessingResult(
            image=processed,
            was_deskewed=was_deskewed,
            skew_angle=skew_angle,
            preprocessing_steps=steps,
        )

    def _deskew_image(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect and correct skew angle using Hough line transform.
        Invoices scanned at an angle cause significant OCR degradation.
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough lines to find dominant angle
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return gray, 0.0

        angles = []
        for line in lines[:20]:  # Use top 20 lines only
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 45:  # Ignore near-vertical lines
                angles.append(angle)

        if not angles:
            return gray, 0.0

        skew_angle = float(np.median(angles))

        # Only correct if skew is significant (>0.5 degrees)
        if abs(skew_angle) < 0.5:
            return gray, skew_angle

        h, w = gray.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        deskewed = cv2.warpAffine(
            gray, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return deskewed, skew_angle

    @staticmethod
    def from_pil(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR array."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def to_pil(cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV grayscale/BGR array to PIL Image."""
        if len(cv_image.shape) == 2:
            return Image.fromarray(cv_image)
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
