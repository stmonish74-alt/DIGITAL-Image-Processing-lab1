
import cv2
import numpy as np
from PIL import Image


# Directional line detection masks
LINE_MASKS = {
    "horizontal": np.array([
        [-1, -1, -1],
        [ 2,  2,  2],
        [-1, -1, -1]
    ], dtype=np.float32),

    "vertical": np.array([
        [-1,  2, -1],
        [-1,  2, -1],
        [-1,  2, -1]
    ], dtype=np.float32),

    "diagonal_45": np.array([
        [-1, -1,  2],
        [-1,  2, -1],
        [ 2, -1, -1]
    ], dtype=np.float32),

    "diagonal_135": np.array([
        [ 2, -1, -1],
        [-1,  2, -1],
        [-1, -1,  2]
    ], dtype=np.float32)
}


def detect_directional_lines(gray, mask, percentile=92):
    """Detect line structures in a particular direction."""

    response = cv2.filter2D(
        gray.astype(np.float32),
        cv2.CV_32F,
        mask
    )

    response = np.abs(response)

    threshold = max(
        float(np.percentile(response, percentile)),
        20.0
    )

    _, detected = cv2.threshold(
        response,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return detected.astype(np.uint8)


def process_lane_markings(image):
    """
    Detect horizontal, vertical and diagonal
    line structures in a road image.

    Supports both PIL Images and OpenCV NumPy arrays.
    """

    if image is None:
        raise ValueError("No input image received.")

    # ------------------------------------------------
    # STEP 1: Convert PIL Image to OpenCV image
    # ------------------------------------------------

    if isinstance(image, Image.Image):

        if image.width == 0 or image.height == 0:
            raise ValueError("Invalid input image.")

        rgb_image = np.array(image.convert("RGB"))

        original = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR
        )

    # ------------------------------------------------
    # STEP 2: Handle NumPy/OpenCV images
    # ------------------------------------------------

    elif isinstance(image, np.ndarray):

        if image.size == 0:
            raise ValueError("Invalid input image.")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 2:
            original = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        elif len(image.shape) == 3 and image.shape[2] == 4:
            original = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR
            )

        elif len(image.shape) == 3 and image.shape[2] == 3:
            original = image.copy()

        else:
            raise ValueError("Unsupported image format.")

    else:
        raise TypeError(
            "Input must be a PIL Image or NumPy array."
        )

    # ------------------------------------------------
    # STEP 3: Convert to grayscale
    # ------------------------------------------------

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------------
    # STEP 4: Apply directional line detection
    # ------------------------------------------------

    results = {}

    for direction, mask in LINE_MASKS.items():

        results[direction] = detect_directional_lines(
            gray,
            mask
        )

    # ------------------------------------------------
    # STEP 5: Combine all directional detections
    # ------------------------------------------------

    combined_mask = cv2.bitwise_or(
        results["horizontal"],
        results["vertical"]
    )

    combined_mask = cv2.bitwise_or(
        combined_mask,
        results["diagonal_45"]
    )

    combined_mask = cv2.bitwise_or(
        combined_mask,
        results["diagonal_135"]
    )

    # Remove isolated noise
    kernel = np.ones((3, 3), np.uint8)

    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ------------------------------------------------
    # STEP 6: Highlight detected lines
    # ------------------------------------------------

    overlay = np.zeros_like(original)

    # Red highlights
    overlay[combined_mask > 0] = (0, 0, 255)

    output = cv2.addWeighted(
        original,
        0.8,
        overlay,
        1.0,
        0
    )

    # ------------------------------------------------
    # STEP 7: Return all results
    # ------------------------------------------------

    return {
        "original": original,
        "grayscale": gray,
        "horizontal": results["horizontal"],
        "vertical": results["vertical"],
        "diagonal_45": results["diagonal_45"],
        "diagonal_135": results["diagonal_135"],
        "combined_mask": combined_mask,
        "output": output
    }
