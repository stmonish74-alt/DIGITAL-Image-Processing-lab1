
import numpy as np
import cv2


def to_binary(image, threshold=127):
    """Convert an image into a binary image."""
    if hasattr(image, "convert"):
        image = np.array(image.convert("L"))
    else:
        image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(
        image, threshold, 255, cv2.THRESH_BINARY
    )
    return binary


def morphological_closing(image, kernel_size=5):
    """Apply morphological closing."""
    binary = to_binary(image)

    kernel = np.ones(
        (kernel_size, kernel_size), np.uint8
    )

    return cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )


def fill_internal_holes(image):
    """Fill internal holes in a binary image."""
    binary = to_binary(image)

    height, width = binary.shape
    flood = binary.copy()

    mask = np.zeros(
        (height + 2, width + 2), np.uint8
    )

    cv2.floodFill(flood, mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)

    return cv2.bitwise_or(binary, holes)
