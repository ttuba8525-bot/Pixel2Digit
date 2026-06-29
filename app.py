"""
app.py — Pixel2Digit Flask Backend
===================================
All 6 API routes:
  GET  /              Home / API index
  POST /predict       Predict digit from uploaded image file
  POST /draw          Predict digit from base64 canvas image
  GET  /model-info    ANN architecture & training details
  GET  /metrics       Performance metrics, confusion matrix, wrong predictions
  POST /reset         Clear uploaded files / session state
"""

import os
import io
import re
import base64
import glob
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf

from predict import (
    predict_digit,
    predict_digit_from_array,
    load_model_if_needed,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any frontend

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB upload limit
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def _file_to_base64(path: str) -> str | None:
    """Read a file from disk and return its base64-encoded content."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _error(message: str, code: int = 400):
    return jsonify({"error": message}), code


# ---------------------------------------------------------------------------
# XAI — Rule-based explanation engine
# ---------------------------------------------------------------------------

# Structural descriptors for each digit
_DIGIT_TRAITS = {
    0: ["single closed oval loop", "no interior segments", "smooth continuous curve"],
    1: ["single vertical stroke", "narrow profile", "minimal horizontal extent"],
    2: ["open loop at top", "diagonal stroke to base", "flat horizontal bottom line"],
    3: ["two open arcs on the right", "no enclosed regions", "two bumps facing right"],
    4: ["combination of vertical and horizontal strokes", "angular structure", "open top triangle"],
    5: ["flat top horizontal bar", "curved bottom loop", "open at top-right"],
    6: ["closed bottom loop", "curved upper tail", "single enclosed region at bottom"],
    7: ["single diagonal downward stroke", "horizontal top bar", "no enclosed regions"],
    8: ["two vertically stacked enclosed loops", "symmetric bilateral structure", "two enclosed regions"],
    9: ["closed upper loop", "descending tail", "single enclosed region at top"],
}

# Why a predicted digit is NOT a specific other digit
_WHY_NOT = {
    # key: (predicted, rival) → reason rival was rejected
    (0, 6): "No tail descending below the oval; the loop is fully symmetric",
    (0, 8): "Only one enclosed region, not two stacked loops",
    (0, 9): "The loop extends equally above and below; no descending tail",
    (1, 7): "No horizontal top bar; only a single vertical stroke",
    (2, 3): "Bottom stroke is flat and closed, not an open arc facing right",
    (2, 7): "Contains a curved top loop, unlike the straight diagonal of 7",
    (3, 2): "Both arcs face right and are open; no flat bottom horizontal bar",
    (3, 8): "Neither arc forms a completely closed loop",
    (4, 9): "No enclosed top loop; the structure is angular and open",
    (5, 6): "The bottom loop is not fully closed; top bar is flat",
    (5, 9): "The curve opens at the top-right, unlike the closed top loop of 9",
    (6, 0): "Has a clear descending tail above the closed bottom loop",
    (6, 5): "The bottom loop is fully enclosed",
    (6, 9): "The enclosed region is at the bottom, not the top",
    (7, 1): "Has a distinct horizontal stroke at the top",
    (8, 0): "Contains two clearly stacked enclosed loops, not one",
    (8, 3): "Both loops are fully closed; no open arcs",
    (8, 9): "Both the upper and lower loops are closed and symmetric",
    (9, 4): "Has a clearly enclosed upper loop; structure is rounded at top",
    (9, 6): "The enclosed region is at the top; the tail descends below",
    (9, 8): "Only the upper loop is closed; the lower portion is an open tail",
}

_DEFAULT_WHY_NOT = "The pixel distribution and structural features do not match the learned pattern for digit {rival}"


def _build_xai(predicted_digit: int, all_probabilities: list, confidence: float) -> dict:
    """
    Builds XAI explanations:
      why_this_digit  – textual reasons why the model chose this class
      why_not_others  – reasons for the top-2 competing classes being rejected
    """
    traits = _DIGIT_TRAITS.get(predicted_digit, ["distinctive stroke patterns"])

    if confidence >= 95:
        confidence_phrase = "Very high confidence"
    elif confidence >= 80:
        confidence_phrase = "High confidence"
    elif confidence >= 60:
        confidence_phrase = "Moderate confidence"
    else:
        confidence_phrase = "Low confidence"

    why_this = (
        f"Prediction: {predicted_digit}\n"
        + "\n".join(f"• {t.capitalize()}" for t in traits)
        + f"\n• {confidence_phrase} ({confidence}%) compared to other classes"
    )

    # Find top-2 rivals (excluding predicted digit)
    sorted_probs = sorted(all_probabilities, key=lambda x: x['probability'], reverse=True)
    rivals = [p for p in sorted_probs if p['digit'] != predicted_digit][:2]

    why_not = []
    for rival_info in rivals:
        rival = rival_info['digit']
        reason = _WHY_NOT.get(
            (predicted_digit, rival),
            _DEFAULT_WHY_NOT.format(rival=rival)
        )
        why_not.append({
            "digit": rival,
            "probability": rival_info['probability'],
            "reason": reason,
        })

    return {
        "why_this_digit": why_this,
        "why_not_others": why_not,
    }


def _build_prediction_journey(steps_completed: int = 5) -> list:
    """Returns the full prediction pipeline journey with status flags."""
    steps = [
        "Uploading",
        "Preprocessing",
        "Running ANN",
        "Generating Prediction",
        "Completed",
    ]
    return [
        {"step": s, "status": "done" if i < steps_completed else "pending"}
        for i, s in enumerate(steps)
    ]


def _enrich_response(raw: dict) -> dict:
    """
    Takes the raw output from predict_digit / predict_digit_from_array and
    adds the XAI layer + prediction journey.
    """
    xai = _build_xai(raw['digit'], raw['all_probabilities'], raw['confidence'])
    journey = _build_prediction_journey()

    return {
        "prediction": {
            "digit": raw['digit'],
            "confidence": raw['confidence'],
            "prediction_time": raw['prediction_time'],
        },
        "all_probabilities": raw['all_probabilities'],
        "top3": raw['top3'],
        "preprocessing_steps": raw['preprocessing_steps'],
        "pixel_grid": raw['pixel_grid'],
        "xai": xai,
        "prediction_journey": journey,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _parse_classification_report(filepath: str) -> tuple[float, float, float, float, list]:
    """
    Parses the saved classification_report.txt and returns
    (accuracy, precision, recall, f1, per_class_list).
    """
    per_class = []
    accuracy  = precision = recall = f1 = 0.0

    if not os.path.exists(filepath):
        return accuracy, precision, recall, f1, per_class

    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    for line in lines:
        # Per-digit lines look like:  "           0     0.9828  0.9929  0.9878   980"
        m = re.match(r'^\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+', line)
        if m:
            per_class.append({
                "digit":     int(m.group(1)),
                "precision": round(float(m.group(2)) * 100, 2),
                "recall":    round(float(m.group(3)) * 100, 2),
                "f1":        round(float(m.group(4)) * 100, 2),
            })

        # accuracy line
        am = re.match(r'^\s+accuracy\s+([\d.]+)\s+\d+', line)
        if am:
            accuracy = round(float(am.group(1)) * 100, 2)

        # macro avg line
        mm = re.match(r'^\s+macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if mm:
            precision = round(float(mm.group(1)) * 100, 2)
            recall    = round(float(mm.group(2)) * 100, 2)
            f1        = round(float(mm.group(3)) * 100, 2)

    return accuracy, precision, recall, f1, per_class


def _get_wrong_predictions(max_samples: int = 10) -> list:
    """
    Loads MNIST test data, runs inference on a sample, and returns up to
    max_samples misclassified examples as base64 images.
    """
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test_norm = x_test.astype('float32') / 255.0
        model = load_model_if_needed()

        # Predict in batches for the full test set (10 000 images)
        preds = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        wrong_indices = np.where(preds != y_test)[0]

        # Sample up to max_samples wrong predictions
        sampled = wrong_indices[:max_samples]
        results = []
        for idx in sampled:
            img_u8 = x_test[idx]                      # uint8 28×28
            img_big = _upscale_for_preview(img_u8)     # 280×280 for clarity
            b64 = _ndarray_to_base64_png(img_big)
            results.append({
                "true_label":      int(y_test[idx]),
                "predicted_label": int(preds[idx]),
                "confidence":      round(
                    float(model.predict(
                        np.expand_dims(x_test_norm[idx], axis=0), verbose=0
                    )[0][preds[idx]]) * 100, 2
                ),
                "image": b64,
            })
        return results
    except Exception as e:
        print(f"[WARN] Could not generate wrong predictions: {e}")
        return []


def _upscale_for_preview(arr_28x28: np.ndarray, scale: int = 10) -> np.ndarray:
    """Upscales a 28×28 uint8 array using nearest-neighbor."""
    return cv2.resize(arr_28x28, (28 * scale, 28 * scale),
                      interpolation=cv2.INTER_NEAREST)


def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# Route 1 — GET / (Serve Dashboard Frontend)
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    """Returns the visual dashboard interface."""
    return render_template('index.html')


# Route 1B — GET /api-info (Serve API Specifications)
# ---------------------------------------------------------------------------
@app.route('/api-info', methods=['GET'])
def api_info():
    """Returns a welcome message and a description of all available endpoints."""
    return jsonify({
        "project": "Pixel2Digit — Explainable Handwritten Digit Recognition",
        "version": "1.0.0",
        "description": (
            "An Explainable AI (XAI) REST API that predicts handwritten digits "
            "using an ANN trained on MNIST, and provides preprocessing "
            "visualizations, probability distributions, and human-readable "
            "decision explanations."
        ),
        "endpoints": [
            {
                "route": "GET /",
                "description": "The NeuroVision XAI Dashboard Frontend"
            },
            {
                "route": "GET /api-info",
                "description": "This specifications page"
            },
            {
                "route": "POST /predict",
                "description": "Predict digit from uploaded image (multipart/form-data, field: 'image')",
                "content_type": "multipart/form-data"
            },
            {
                "route": "POST /draw",
                "description": "Predict digit from canvas drawing (JSON body: {'image': '<base64 PNG>'})",
                "content_type": "application/json"
            },
            {
                "route": "GET /model-info",
                "description": "ANN architecture, optimizer, loss, and parameter counts"
            },
            {
                "route": "GET /metrics",
                "description": "Accuracy, Precision, Recall, F1, confusion matrix, and wrong predictions"
            },
            {
                "route": "POST /reset",
                "description": "Clear all uploaded images from the server"
            },
        ],
        "model_ready": os.path.exists(os.path.join(MODEL_DIR, 'model.keras')),
    })


# ---------------------------------------------------------------------------
# Route 2 — POST /predict
# ---------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a handwritten digit image via multipart/form-data (field name: 'image').

    Returns full XAI-enriched prediction response including:
      - digit, confidence, prediction_time
      - all_probabilities (0-9)
      - top3
      - preprocessing_steps (original, grayscale, resized, normalized) as base64 PNGs
      - pixel_grid (28×28 float values)
      - xai (why_this_digit, why_not_others)
      - prediction_journey (pipeline step statuses)
    """
    if 'image' not in request.files:
        return _error("No 'image' field found in the form data.")

    file = request.files['image']

    if file.filename == '':
        return _error("No file selected.")

    if not _allowed_file(file.filename):
        return _error(
            f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        raw = predict_digit(save_path)
        return jsonify(_enrich_response(raw))
    except Exception as e:
        return _error(f"Prediction failed: {str(e)}", 500)


# ---------------------------------------------------------------------------
# Route 3 — POST /draw
# ---------------------------------------------------------------------------
@app.route('/draw', methods=['POST'])
def draw():
    """
    Accepts a base64-encoded PNG of a canvas drawing.

    Expected JSON body:
      { "image": "data:image/png;base64,<base64 string>" }
      OR
      { "image": "<raw base64 string without data URI prefix>" }

    Returns the same enriched response as /predict.
    """
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return _error("JSON body must contain an 'image' key with a base64-encoded PNG.")

    b64_str = data['image']

    # Strip optional "data:image/...;base64," prefix
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGBA')

        # Composite onto black background (handles transparent canvases)
        bg = Image.new('RGBA', pil_img.size, (0, 0, 0, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        img_rgb = bg.convert('RGB')

        # Convert to grayscale NumPy array
        img_gray = np.array(img_rgb.convert('L'))

        # Invert if background is light (canvas drawings are typically
        # dark stroke on white background)
        if np.mean(img_gray) > 127:
            img_gray = 255 - img_gray

        raw = predict_digit_from_array(img_gray)
        return jsonify(_enrich_response(raw))

    except Exception as e:
        return _error(f"Canvas prediction failed: {str(e)}", 500)


# ---------------------------------------------------------------------------
# Route 4 — GET /model-info
# ---------------------------------------------------------------------------
@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Returns detailed ANN architecture information.

    Response includes:
      - architecture name
      - optimizer & loss function
      - input shape
      - per-layer details (name, units/rate, activation)
      - total and trainable parameter counts
    """
    try:
        model = load_model_if_needed()
    except FileNotFoundError as e:
        return _error(str(e), 404)

    layers_info = []
    for layer in model.layers:
        entry = {"name": type(layer).__name__}

        try:
            entry['units'] = int(layer.units) if hasattr(layer, 'units') else None
        except Exception:
            entry['units'] = None

        try:
            entry['rate'] = float(layer.rate) if hasattr(layer, 'rate') else None
        except Exception:
            entry['rate'] = None

        try:
            if hasattr(layer, 'activation'):
                act = layer.activation
                entry['activation'] = act.__name__ if callable(act) else str(act)
            else:
                entry['activation'] = None
        except Exception:
            entry['activation'] = None

        # Keras 3 removed layer.output_shape; use output_spec or skip safely
        try:
            entry['output_shape'] = str(layer.output_shape)
        except AttributeError:
            try:
                spec = layer.compute_output_spec(layer.input_spec)
                entry['output_shape'] = str(spec.shape) if hasattr(spec, 'shape') else None
            except Exception:
                entry['output_shape'] = None
        except Exception:
            entry['output_shape'] = None

        layers_info.append(entry)

    total_params     = int(model.count_params())
    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))

    # Reconstruct compile config safely
    try:
        compile_config = model.optimizer.get_config()
        optimizer_name = type(model.optimizer).__name__
    except Exception:
        compile_config = {}
        optimizer_name = "Adam"

    return jsonify({
        "architecture": "Sequential ANN",
        "framework":    "TensorFlow / Keras",
        "optimizer":    optimizer_name,
        "optimizer_config": compile_config,
        "loss_function": "Sparse Categorical Crossentropy",
        "metrics":       ["accuracy"],
        "input_shape":   [784],
        "output_classes": 10,
        "layers":        layers_info,
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "dataset": "MNIST (60,000 train / 10,000 test)",
        "training": {
            "epochs":            15,
            "batch_size":        64,
            "validation_split":  0.1,
            "early_stopping":    True,
            "checkpoint":        True,
        },
    })


# ---------------------------------------------------------------------------
# Route 5 — GET /metrics
# ---------------------------------------------------------------------------
@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Returns model performance metrics, confusion matrix, and wrong predictions.

    Query parameters:
      wrong_predictions (bool, default true) – include wrong prediction gallery
      max_wrong        (int,  default 10)    – max number of wrong samples to return

    Response includes:
      - accuracy, precision, recall, f1_score (macro)
      - per_class breakdown for each digit (0-9)
      - confusion_matrix (10×10 2D int array)
      - confusion_matrix_image (base64 PNG)
      - wrong_predictions (gallery of misclassified test samples)
    """
    include_wrong = request.args.get('wrong_predictions', 'true').lower() != 'false'
    max_wrong     = int(request.args.get('max_wrong', 10))

    report_path = os.path.join(BASE_DIR, 'classification_report.txt')
    cm_path     = os.path.join(BASE_DIR, 'confusion_matrix.png')

    accuracy, precision, recall, f1, per_class = _parse_classification_report(report_path)

    # Confusion matrix image as base64
    cm_image_b64 = _file_to_base64(cm_path)

    # Rebuild confusion matrix as a 2D array from the trained model + MNIST test set
    confusion_matrix_2d = None
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test_norm = x_test.astype('float32') / 255.0
        model   = load_model_if_needed()
        y_pred  = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_test, y_pred)
        confusion_matrix_2d = cm.tolist()
    except Exception as e:
        print(f"[WARN] Could not build confusion matrix array: {e}")

    wrong_predictions = []
    if include_wrong:
        wrong_predictions = _get_wrong_predictions(max_wrong)

    return jsonify({
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1_score":   f1,
        "per_class":  per_class,
        "confusion_matrix":       confusion_matrix_2d,
        "confusion_matrix_image": cm_image_b64,
        "wrong_predictions":      wrong_predictions,
        "report_generated_from":  report_path,
    })


# ---------------------------------------------------------------------------
# Route 6 — POST /reset
# ---------------------------------------------------------------------------
@app.route('/reset', methods=['POST'])
def reset():
    """
    Clears all uploaded images from the server's uploads/ directory.

    Returns:
      { "status": "ok", "message": "...", "files_removed": N }
    """
    try:
        files = glob.glob(os.path.join(UPLOAD_DIR, '*'))
        count = 0
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
                count += 1
        return jsonify({
            "status":        "ok",
            "message":       f"Upload state cleared. {count} file(s) removed.",
            "files_removed": count,
        })
    except Exception as e:
        return _error(f"Reset failed: {str(e)}", 500)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "hint": "Try GET / for a list of routes."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 10 MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error.", "details": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pixel2Digit Flask Backend')
    parser.add_argument('--host', default='0.0.0.0',  help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', default=5000, type=int, help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    print("=" * 60)
    print("  Pixel2Digit — Explainable Digit Recognition API")
    print(f"  Running on http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)
