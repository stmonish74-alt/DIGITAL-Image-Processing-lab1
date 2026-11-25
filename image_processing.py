
import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError


def decode_data_url(data_url):

    if not data_url:
        raise ValueError("Please upload an image.")

    if not isinstance(data_url, str):
        raise ValueError("Invalid image data.")

    if "," in data_url:
        header, data_url = data_url.split(",", 1)

        if not header.startswith("data:image/") or ";base64" not in header:
            raise ValueError("Invalid image format.")

    if not data_url.strip():
        raise ValueError("Image data is empty.")

    try:
        image_bytes = base64.b64decode(
            data_url,
            validate=True
        )

        if not image_bytes:
            raise ValueError("Image data is empty.")

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "Invalid or corrupted Base64 image data."
        ) from exc

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "Unable to read the uploaded image."
        ) from exc


def convert_to_pil(image):

    if image is None:
        raise ValueError("No processed image was provided.")

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):

        if image.size == 0:
            raise ValueError("Processed image is empty.")

        if image.dtype != np.uint8:
            image = np.clip(
                image, 0, 255
            ).astype(np.uint8)

        if image.ndim == 2:
            return Image.fromarray(image)

        if image.ndim == 3:

            if image.shape[2] == 3:
                return Image.fromarray(image, mode="RGB")

            if image.shape[2] == 4:
                return Image.fromarray(image, mode="RGBA")

        raise ValueError("Unsupported image dimensions.")

    raise ValueError("Invalid processed image type.")


def encode_data_url(image):

    try:
        pil_image = convert_to_pil(image)

        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")

        buffer = io.BytesIO()

        pil_image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return "data:image/png;base64," + encoded

    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
