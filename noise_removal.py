from PIL import Image, ImageFilter


def apply_filter(image, method="median", kernel=3):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    kernel = int(kernel)

    if kernel < 3 or kernel > 15:
        raise ValueError("Kernel size must be between 3 and 15.")

    if kernel % 2 == 0:
        kernel += 1

    if method == "median":
        return image.filter(ImageFilter.MedianFilter(size=kernel))

    elif method == "gaussian":
        return image.filter(ImageFilter.GaussianBlur(radius=kernel / 3))

    elif method == "mean":
        return image.filter(ImageFilter.BoxBlur(radius=kernel / 2))

    else:
        raise ValueError("Unsupported filter method.")