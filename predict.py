import os
import io
import time
import base64
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Global variable to cache the loaded model
_model = None


def load_model_if_needed():
    """
    Lazy-loads the Keras model to memory.  Thread-safe; prevents the module
    import from failing if the model has not been trained yet.
    """
    global _model
    if _model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'model.keras')

        if not os.path.exists(model_path):
            model_path = os.path.join('model', 'model.keras')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Please train the model first by running: python train_model.py"
            )

        print(f"Loading model from: {model_path}...")
        _model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    """
    Converts a NumPy image array (uint8, grayscale or RGB) to a base64-encoded
    PNG string suitable for embedding in JSON.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        pil_img = Image.fromarray(arr, mode='L')
    else:
        pil_img = Image.fromarray(arr, mode='RGB')

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _upscale_for_preview(arr_28x28: np.ndarray, scale: int = 10) -> np.ndarray:
    """
    Upscales a 28x28 uint8 array to (28*scale)x(28*scale) using nearest-neighbor
    so individual pixels are clearly visible in the preview.
    """
    return cv2.resize(arr_28x28, (28 * scale, 28 * scale),
                      interpolation=cv2.INTER_NEAREST)


def center_digit(img: np.ndarray) -> np.ndarray:
    """
    Centers the digit in the image matching the MNIST preprocessing rules:
    - Adaptively normalizes stroke thickness in high-res space to match MNIST.
    - Finds the bounding box of the digit.
    - Crops the digit.
    - Resizes the cropped digit to fit in a 20x20 box (preserving aspect ratio).
    - Centers the 20x20 digit inside a 28x28 blank (black) canvas.
    """
    # Threshold to identify active pixels
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    active_area = np.sum(thresh > 0)
    
    # Adaptively dilate thin strokes in 280x280 space
    if active_area > 0:
        if active_area < 9000:
            k_size = 25
        elif active_area < 16000:
            k_size = 17
        elif active_area < 24000:
            k_size = 9
        else:
            k_size = 0
            
        if k_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            img = cv2.dilate(img, kernel, iterations=1)

    # Re-threshold to get bounding box on dilated image
    _, thresh_dilated = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh_dilated)
    if coords is None:
        return cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    
    if w > h:
        new_w = 20
        new_h = int(h * (20.0 / w))
    else:
        new_h = 20
        new_w = int(w * (20.0 / h))
        
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    resized_digit = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create blank 28x28 image
    centered_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Center offsets
    dx = (28 - new_w) // 2
    dy = (28 - new_h) // 2
    
    centered_img[dy:dy+new_h, dx:dx+new_w] = resized_digit
    return centered_img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_digit(image_path: str) -> dict:
    """
    Predicts the handwritten digit from a saved image file.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        dict with keys:
            digit               – predicted class (int 0-9)
            confidence          – probability of predicted class in % (float)
            top3                – list of {digit, probability} for top 3 classes
            all_probabilities   – list of {digit, probability} for all 10 classes
            prediction_time     – human-readable string, e.g. "42 ms"
            preprocessing_steps – dict of {original, grayscale, resized,
                                  normalized} as base64 PNG strings
            pixel_grid          – {width, height, pixels} where pixels is a
                                  28×28 list-of-lists of float values [0,1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No image found at path: {image_path}")

    # ------------------------------------------------------------------
    # Step 1 – Read original image (BGR)
    # ------------------------------------------------------------------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"OpenCV failed to read the image at: {image_path}")

    # Save original as RGB preview
    img_original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    b64_original = _ndarray_to_base64_png(img_original_rgb)

    # ------------------------------------------------------------------
    # Step 2 – Convert to grayscale
    # ------------------------------------------------------------------
    start_time = time.time()

    if len(img_bgr.shape) == 3:
        if img_bgr.shape[2] == 4:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr.copy()

    b64_grayscale = _ndarray_to_base64_png(img_gray)

    # ------------------------------------------------------------------
    # Step 3 – Invert if background is light (MNIST style: white digit on black)
    # ------------------------------------------------------------------
    mean_intensity = np.mean(img_gray)
    if mean_intensity > 127:
        img_gray = 255 - img_gray

    # ------------------------------------------------------------------
    # Step 4 – Center and Resize to 28×28
    # ------------------------------------------------------------------
    img_resized = center_digit(img_gray)
    # Preview: upscale so the pixel grid is visible
    b64_resized = _ndarray_to_base64_png(_upscale_for_preview(img_resized))

    # ------------------------------------------------------------------
    # Step 5 – Normalize to [0, 1]
    # ------------------------------------------------------------------
    img_normalized = img_resized.astype('float32') / 255.0
    # Convert back to uint8 for visualization (multiply by 255)
    img_normalized_u8 = (img_normalized * 255).astype(np.uint8)
    b64_normalized = _ndarray_to_base64_png(_upscale_for_preview(img_normalized_u8))

    # ------------------------------------------------------------------
    # Step 6 – Flatten and predict
    # ------------------------------------------------------------------
    img_input = np.expand_dims(img_normalized, axis=0)   # shape (1, 28, 28)

    model = load_model_if_needed()
    probabilities = model.predict(img_input, verbose=0)[0]  # shape (10,)

    end_time = time.time()
    prediction_time_ms = int(round((end_time - start_time) * 1000))

    # ------------------------------------------------------------------
    # Step 7 – Build result structures
    # ------------------------------------------------------------------
    # All 10 probabilities
    all_probs = [
        {"digit": int(i), "probability": float(round(float(probabilities[i]) * 100, 2))}
        for i in range(10)
    ]

    # Top-3
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {"digit": int(idx), "probability": float(round(float(probabilities[idx]) * 100, 2))}
        for idx in top3_indices
    ]

    predicted_digit = int(top3_indices[0])
    confidence = float(round(float(probabilities[predicted_digit]) * 100, 2))

    # 28×28 pixel grid as nested float list
    pixel_grid = {
        "width": 28,
        "height": 28,
        "pixels": img_normalized.tolist()
    }

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "top3": top3,
        "all_probabilities": all_probs,
        "prediction_time": f"{prediction_time_ms} ms",
        "preprocessing_steps": {
            "original":   b64_original,
            "grayscale":  b64_grayscale,
            "resized":    b64_resized,
            "normalized": b64_normalized,
        },
        "pixel_grid": pixel_grid,
    }


def predict_digit_from_array(img_gray_28x28: np.ndarray) -> dict:
    """
    Predicts a digit from a pre-processed 28×28 grayscale NumPy array.
    Used internally by the /draw route (canvas data arrives already cropped).

    The array should be uint8 in [0, 255] with white digit on black background.

    Returns the same dict shape as predict_digit().
    """
    start_time = time.time()

    img_resized = center_digit(img_gray_28x28)
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    model = load_model_if_needed()
    probabilities = model.predict(img_input, verbose=0)[0]

    end_time = time.time()
    prediction_time_ms = int(round((end_time - start_time) * 1000))

    all_probs = [
        {"digit": int(i), "probability": float(round(float(probabilities[i]) * 100, 2))}
        for i in range(10)
    ]

    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {"digit": int(idx), "probability": float(round(float(probabilities[idx]) * 100, 2))}
        for idx in top3_indices
    ]

    predicted_digit = int(top3_indices[0])
    confidence = float(round(float(probabilities[predicted_digit]) * 100, 2))

    # Preprocessing steps (for canvas input we only have resized + normalized)
    b64_resized   = _ndarray_to_base64_png(_upscale_for_preview(img_resized))
    img_norm_u8   = (img_normalized * 255).astype(np.uint8)
    b64_normalized = _ndarray_to_base64_png(_upscale_for_preview(img_norm_u8))

    pixel_grid = {
        "width": 28,
        "height": 28,
        "pixels": img_normalized.tolist()
    }

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "top3": top3,
        "all_probabilities": all_probs,
        "prediction_time": f"{prediction_time_ms} ms",
        "preprocessing_steps": {
            "original":   None,   # canvas has no "original colour" step
            "grayscale":  None,
            "resized":    b64_resized,
            "normalized": b64_normalized,
        },
        "pixel_grid": pixel_grid,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    try:
        result = predict_digit(sys.argv[1])
        # Print without the large base64 blobs for readability
        printable = {k: v for k, v in result.items()
                     if k not in ('preprocessing_steps', 'pixel_grid')}
        print("\nPrediction Result:")
        print(json.dumps(printable, indent=2))
    except Exception as err:
        print(f"Execution Error: {err}")
        sys.exit(1)
