import cv2
import numpy as np
from functools import lru_cache

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@lru_cache(maxsize=1)
def load_models():
    """Train the demo classifiers once per Flask process and cache them."""
    dataset = load_digits()
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma=0.001, probability=True, random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, predicted) * 100), 2),
            "precision": round(float(precision_score(y_test, predicted, average="weighted", zero_division=0) * 100), 2),
            "recall": round(float(recall_score(y_test, predicted, average="weighted", zero_division=0) * 100), 2),
            "f1": round(float(f1_score(y_test, predicted, average="weighted", zero_division=0) * 100), 2),
        }
    return models, metrics


def prepare_digit_image(image):
    """Convert a single handwritten digit image to 64 Digits-dataset features."""
    if image is None:
        raise ValueError("Please upload an image.")

    if hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)

    if image.size == 0:
        raise ValueError("Uploaded image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format. Please upload a PNG or JPEG image.")

    # Smooth isolated pixel noise without eroding thin handwritten strokes.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Keep the foreground digit white on a black background.
    border = np.concatenate((binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]))
    if float(np.mean(border)) > 127:
        binary = cv2.bitwise_not(binary)

    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("No digit detected. Use a clear image with a visible digit.")

    x, y, width, height = cv2.boundingRect(points)
    cropped = binary[y:y + height, x:x + width]

    # Fit the digit inside a 20x20 area, preserving its aspect ratio.
    scale = 20.0 / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    # Align the foreground's center of mass near the image center.
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, transform, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)

    # sklearn Digits images use 8x8 pixels with values in the 0–16 range.
    small = cv2.resize(canvas, (8, 8), interpolation=cv2.INTER_AREA)
    features = np.clip((small.astype(np.float32) / 255.0) * 16.0, 0, 16)
    return canvas, features.reshape(1, 64)


def recognize_digit(image):
    processed_image, features = prepare_digit_image(image)
    models, metrics = load_models()
    predictions = {}

    for name, model in models.items():
        predicted_digit = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        predictions[name] = {
            "digit": predicted_digit,
            "confidence": round(float(np.max(probabilities) * 100), 2),
        }

    primary = predictions["SVM"]
    feature_matrix = features.reshape(8, 8)
    feature_values = {
        "pixel_intensity": [round(float(v), 2) for v in feature_matrix.reshape(-1)],
        "horizontal_projection": [round(float(v), 2) for v in feature_matrix.sum(axis=1)],
        "vertical_projection": [round(float(v), 2) for v in feature_matrix.sum(axis=0)],
    }
    display_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

    return {
        "digit": primary["digit"],
        "confidence": primary["confidence"],
        "predictions": predictions,
        "metrics": metrics,
        "dataset": "Scikit-learn Optical Recognition of Handwritten Digits Dataset",
        "method": "Image preprocessing with SVM, Random Forest and KNN",
        "feature_values": feature_values,
        "processed_image": display_image,
    }
