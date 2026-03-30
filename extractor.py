Tesseract OCR engine wrapper + regex-based field extraction.
Extracts Vendor Name, GST Number, Total Amount, and Invoice Date
from preprocessed invoice images.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import pytesseract
import numpy as np
from PIL import Image

from src.ocr.preprocessor import ImagePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for Indian invoice fields
# ---------------------------------------------------------------------------

# GST: 2-digit state code + 10-char PAN + 1-digit entity + Z + 1 checksum
GST_PATTERN = re.compile(
    r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b"
)

# Total Amount: handles ₹, Rs., INR prefixes and comma-formatted numbers
AMOUNT_PATTERNS = [
    re.compile(r"(?:total\s+amount|grand\s+total|total\s+due|amount\s+payable)"
               r"\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+\.?\d*)\s*(?:only|/-)?", re.IGNORECASE),
]

# Invoice Date
DATE_PATTERNS = [
    re.compile(r"(?:invoice\s+date|date\s+of\s+invoice|date)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b"),
]

# Vendor name usually appears near "from:", "vendor:", "billed by:", "supplier:"
VENDOR_PATTERNS = [
    re.compile(r"(?:from|vendor|supplier|billed\s+by|sold\s+by)[:\s]+([A-Z][A-Za-z\s&.,'-]{3,50})", re.IGNORECASE),
    re.compile(r"^([A-Z][A-Z\s&.,'-]{5,50}(?:pvt\.?\s*ltd\.?|ltd\.?|inc\.?|llp|llc))", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class ExtractionResult:
    """Holds all extracted fields plus OCR metadata."""
    raw_text: str
    vendor_name: Optional[str] = None
    gst_number: Optional[str] = None
    total_amount: Optional[float] = None
    invoice_date: Optional[str] = None

    # Confidence scores (0–100, from Tesseract word-level confidence)
    ocr_confidence: float = 0.0

    # Field-level confidence flags
    vendor_confidence: str = "low"    # "high" | "medium" | "low"
    gst_confidence: str = "low"
    amount_confidence: str = "low"

    # Processing metadata
    preprocessing_steps: list[str] = field(default_factory=list)
    extraction_warnings: list[str] = field(default_factory=list)


class InvoiceExtractor:
    """
    Full extraction pipeline:
        preprocessed image → Tesseract OCR → regex field extraction → ExtractionResult
    """

    def __init__(
        self,
        tesseract_config: str = "--oem 3 --psm 6",
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        """
        Args:
            tesseract_config: Tesseract page segmentation mode.
                --oem 3: Use LSTM neural net OCR engine (most accurate)
                --psm 6: Assume a single uniform block of text (good for invoices)
            preprocessor: ImagePreprocessor instance (uses defaults if None)
        """
        self.tesseract_config = tesseract_config
        self.preprocessor = preprocessor or ImagePreprocessor()

    def extract_from_image(self, image: np.ndarray) -> ExtractionResult:
        """
        Run full extraction pipeline on a single OpenCV image array.

        Args:
            image: BGR numpy array

        Returns:
            ExtractionResult with all fields and confidence scores
        """
        # Step 1: Preprocess
        prep_result: PreprocessingResult = self.preprocessor.process(image)

        # Step 2: Tesseract OCR — get both text and confidence data
        pil_image = ImagePreprocessor.to_pil(prep_result.image)
        raw_text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
        ocr_confidence = self._get_mean_confidence(pil_image)

        logger.debug(f"OCR confidence: {ocr_confidence:.1f}%")
        logger.debug(f"Raw text (first 200 chars): {raw_text[:200]}")

        # Step 3: Field extraction
        result = ExtractionResult(
            raw_text=raw_text,
            ocr_confidence=ocr_confidence,
            preprocessing_steps=prep_result.preprocessing_steps,
        )

        result.gst_number, result.gst_confidence = self._extract_gst(raw_text)
        result.total_amount, result.amount_confidence = self._extract_amount(raw_text)
        result.vendor_name, result.vendor_confidence = self._extract_vendor(raw_text)
        result.invoice_date = self._extract_date(raw_text)

        # Step 4: Warnings for low-confidence fields
        if ocr_confidence < 60:
            result.extraction_warnings.append(
                f"Low overall OCR confidence ({ocr_confidence:.1f}%) — "
                "image quality may be poor"
            )
        for field_name, conf in [
            ("vendor_name", result.vendor_confidence),
            ("gst_number", result.gst_confidence),
            ("total_amount", result.amount_confidence),
        ]:
            if conf == "low":
                result.extraction_warnings.append(
                    f"Field '{field_name}' extracted with low confidence — "
                    "manual review recommended"
                )

        return result

    # ------------------------------------------------------------------
    # Private field extractors
    # ------------------------------------------------------------------

    def _extract_gst(self, text: str) -> tuple[Optional[str], str]:
        match = GST_PATTERN.search(text)
        if match:
            gst = match.group(1).upper()
            return gst, "high"  # Regex match on a fixed-format field = high confidence
        return None, "low"

    def _extract_amount(self, text: str) -> tuple[Optional[float], str]:
        for i, pattern in enumerate(AMOUNT_PATTERNS):
            match = pattern.search(text)
            if match:
                raw = match.group(1).replace(",", "")
                try:
                    amount = float(raw)
                    confidence = "high" if i == 0 else "medium"
                    return amount, confidence
                except ValueError:
                    continue
        return None, "low"

    def _extract_vendor(self, text: str) -> tuple[Optional[str], str]:
        for i, pattern in enumerate(VENDOR_PATTERNS):
            match = pattern.search(text)
            if match:
                vendor = " ".join(match.group(1).split())  # normalize whitespace
                confidence = "high" if i == 0 else "medium"
                return vendor[:100], confidence  # cap at 100 chars
        return None, "low"

    def _extract_date(self, text: str) -> Optional[str]:
        for pattern in DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _get_mean_confidence(self, pil_image: Image.Image) -> float:
        """
        Use Tesseract's word-level confidence scores to compute mean confidence.
        This is a better signal than just checking if text was extracted.
        """
        try:
            data = pytesseract.image_to_data(
                pil_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                int(c) for c in data["conf"]
                if str(c).strip() not in ("-1", "") and int(c) >= 0
            ]
            return float(np.mean(confidences)) if confidences else 0.0
        except Exception as e:
            logger.warning(f"Could not compute confidence scores: {e}")
            return 0.0
