# NeuroVision XAI Lab - Frontend System

Explainable Handwritten Digit Recognition using Artificial Neural Networks (ANN) web portal. This repository contains the complete responsive user interface built using HTML5, CSS3, and Vanilla JavaScript, with dynamic visual data streams powered by Chart.js.

## Folder Structure

```text
DigitRecognition/
│
├── templates/
│   └── index.html      # Glassmorphism layout & SVG network dashboard
│
├── static/
│   ├── style.css       # Dark-futuristic styles, neon glows, & animations
│   └── script.js       # Canvas drawings, particle systems, & fallbacks
│
└── README.md           # Instructions and server configurations
```

---

## Running the Project

To run this application as a local Flask site, follow the simple setup below:

### 1. Prerequisite Installations
Ensure you have Python installed, then install Flask:
```bash
pip install Flask
```

### 2. Create the Backend Server (`app.py`)
In your project root directory (alongside `templates/` and `static/`), create a file named `app.py` and paste the following starter template:

```python
from flask import Flask, render_code, render_template, jsonify, request
import time
import random

app = Flask(__name__)

# Route to serve the single-page dashboard
@app.route('/')
def home():
    return render_template('index.html')

# POST: Predict uploaded image files
@app.route('/predict', methods=['POST'])
def predict():
    # Simulate processing delay
    time.sleep(0.1)
    
    # Placeholder return matching frontend expectations
    # (Replace this with your trained Keras/TensorFlow model inference)
    predicted_digit = random.randint(0, 9)
    confidence = float(random.uniform(92.0, 99.8))
    probabilities = [random.uniform(0.0, 10.0) for _ in range(10)]
    probabilities[predicted_digit] = confidence
    
    # Normalize probabilities to sum close to 100%
    probs_sum = sum(probabilities)
    probabilities = [(p / probs_sum) * 100 for p in probabilities]
    
    top_predictions = sorted(
        [{"digit": i, "probability": p} for i, p in enumerate(probabilities)],
        key=lambda x: x["probability"],
        reverse=True
    )
    
    return jsonify({
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "probabilities": probabilities,
        "processing_time": int(random.uniform(35, 120)),
        "explanation": f"The model predicted {predicted_digit} because the digit contains shape properties matching that class.",
        "why_not": [
            {"digit": (predicted_digit + 1) % 10, "reason": "The contour angles do not match activation layers."},
            {"digit": (predicted_digit + 2) % 10, "reason": "Central stroke densities are below the activation threshold."}
        ],
        "preprocessing": {
            "original": request.form.get("image", ""),
            "grayscale": "",
            "resized": "",
            "normalized": "",
            "pixel_grid": [0.0] * 784
        }
    })

# POST: Predict drawn coordinates from canvas
@app.route('/draw', methods=['POST'])
def draw():
    # Stub coordinates prediction matching /predict format
    return predict()

# GET: Return model configuration
@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "architecture": ["Input 784", "Dense 256", "Dense 128", "Dropout", "Dense 64", "Output 10"],
        "optimizer": "Adam",
        "loss_function": "Categorical Crossentropy",
        "accuracy": 98.50
    })

# GET: Return overall model performance stats
@app.route('/metrics', methods=['GET'])
def metrics():
    # Generate 10x10 mock confusion matrix for visualization
    matrix = [[random.randint(920, 995) if r == c else random.randint(0, 15) for c in range(10)] for r in range(10)]
    return jsonify({
        "accuracy": 98.50,
        "precision": 98.20,
        "recall": 98.10,
        "f1_score": 98.15,
        "confusion_matrix": matrix,
        "wrong_predictions": [
            {
                "image_url": "",
                "actual_digit": 4,
                "predicted_digit": 9,
                "confidence": 76.5,
                "explanation": "High horizontal stroke closed the top gap."
            }
        ]
    })

# POST: Reset server session cache
@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({"status": "success", "message": "System logs refreshed."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 3. Launch the Server
Run the python app from your console:
```bash
python app.py
```

### 4. View in Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## Dynamic Self-Healing Offline Mode

If the backend server is offline or is not running, **script.js** is equipped with a **self-healing fallback mechanism**:
*   The application detects fetch connection failures and automatically transitions to **mock execution**.
*   When you draw on the canvas or drop an image, a hidden rendering thread scales your strokes to a real **28x28 grayscale matrix** to populate the preprocessing pixel grids, heatmaps, and probability graphs locally.
*   Hyperparameters, confusion matrix scores, and wrong prediction galleries populate with standard MNIST statistics so that the visual UI works out of the box.
