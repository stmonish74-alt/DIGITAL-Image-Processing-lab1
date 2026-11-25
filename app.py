
from flask import Flask, jsonify, render_template, request

from backend.image_processing import (
    decode_data_url,
    encode_data_url,
)

from backend.noise_removal import apply_filter

from backend.region_filling import (
    to_binary,
    fill_internal_holes,
    morphological_closing,
)

from backend.lane_detection import process_lane_markings

from backend.digit_recognition import recognize_digit


# ==================================================
# FLASK CONFIGURATION
# ==================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():
    return render_template("index.html")


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health():

    return jsonify(
        success=True,
        status="ok",
        message="DIGITAL Image Processing backend is running",
    )


# ==================================================
# SALT AND PEPPER NOISE REMOVAL
# ==================================================

@app.post("/api/noise/filter")
def noise_filter():

    data = request.get_json(silent=True) or {}

    try:

        image = decode_data_url(data.get("image"))

        result = apply_filter(
            image,
            data.get("method", "median"),
            int(data.get("kernel", 3)),
        )

        return jsonify(
            success=True,
            image=encode_data_url(result),
        )

    except (ValueError, TypeError) as exc:

        return jsonify(
            success=False,
            error=str(exc),
        ), 400

    except Exception:

        app.logger.exception("Noise filtering failed")

        return jsonify(
            success=False,
            error="Noise filtering failed. Check the Flask terminal.",
        ), 500


# ==================================================
# REGION FILLING AND MORPHOLOGICAL CLOSING
# ==================================================

@app.post("/api/morph/process")
def morph_process():

    data = request.get_json(silent=True) or {}

    try:

        image = decode_data_url(data.get("image"))

        binary = to_binary(
            image,
            int(data.get("threshold", 128)),
        )

        action = data.get("action", "fill")

        if action == "fill":

            result = fill_internal_holes(binary)

        elif action == "closing":

            result = morphological_closing(
                binary,
                int(data.get("kernel", 3)),
            )

        else:

            return jsonify(
                success=False,
                error="Action must be fill or closing.",
            ), 400

        return jsonify(
            success=True,
            image=encode_data_url(result),
        )

    except (ValueError, TypeError) as exc:

        return jsonify(
            success=False,
            error=str(exc),
        ), 400

    except Exception:

        app.logger.exception("Morphological processing failed")

        return jsonify(
            success=False,
            error="Morphological processing failed. Check the Flask terminal.",
        ), 500


# ==================================================
# ROAD LANE MARKING DETECTION
# ==================================================

@app.post("/api/lane/detect")
def lane_detect():

    data = request.get_json(silent=True) or {}

    try:

        image = decode_data_url(data.get("image"))

        results = process_lane_markings(image)

        encoded_results = {
            name: encode_data_url(result)
            for name, result in results.items()
        }

        return jsonify(
            success=True,
            message="Road lane line detection completed successfully.",
            images=encoded_results,
        )

    except (ValueError, TypeError) as exc:

        return jsonify(
            success=False,
            error=str(exc),
        ), 400

    except Exception:

        app.logger.exception("Lane detection failed")

        return jsonify(
            success=False,
            error="Lane detection failed. Check the Flask terminal.",
        ), 500


# ==================================================
# HANDWRITTEN DIGIT RECOGNITION
# ==================================================

@app.post("/api/digit/recognize")
def digit_recognize():

    data = request.get_json(silent=True) or {}

    try:

        image_data = data.get("image")

        if not image_data:
            return jsonify(
                success=False,
                error="Please upload a handwritten digit image.",
            ), 400

        image = decode_data_url(image_data)

        result = recognize_digit(image)

        processed_image = encode_data_url(
            result.pop("processed_image")
        )

        return jsonify(
            success=True,
            **result,
            processed_image=processed_image,
        )

    except (ValueError, TypeError) as exc:

        app.logger.warning(
            "Digit recognition input error: %s",
            exc,
        )

        return jsonify(
            success=False,
            error=str(exc),
        ), 400

    except Exception:

        app.logger.exception("Digit recognition failed")

        return jsonify(
            success=False,
            error="Digit recognition failed. Check the Flask terminal.",
        ), 500


# ==================================================
# ERROR HANDLERS
# ==================================================

@app.errorhandler(413)
def too_large(_error):

    return jsonify(
        success=False,
        error="Image is too large. Maximum upload size is 12 MB.",
    ), 413


@app.errorhandler(404)
def not_found(error):

    if request.path.startswith("/api/"):

        return jsonify(
            success=False,
            error="API endpoint not found: " + request.path,
        ), 404

    return error


@app.errorhandler(500)
def internal_server_error(error):

    if request.path.startswith("/api/"):

        app.logger.exception(
            "API request failed: %s",
            request.path,
        )

        return jsonify(
            success=False,
            error="A server error occurred. Check the Flask terminal.",
        ), 500

    return error


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=True,
    )
