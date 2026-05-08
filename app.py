"""
DermDx AI — Flask Backend
Model: SqueezeNet1_1 fine-tuned for 9-class skin disease classification
Run:   python app.py
"""

import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow React dev server

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = r"c:\Users\asus\Downloads\SkinDiseaseProject\skin_disease_model.pth"

CLASS_NAMES = [
    "Actinic Keratosis",
    "Atopic Dermatitis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanocytic Nevus",
    "Melanoma",
    "Squamous Cell Carcinoma",
    "Tinea / Ringworm / Candidiasis",
    "Vascular Lesion",
]

# Risk levels — keep in sync with your React DISEASE_CLASSES array
RISK_LEVELS = {
    "Actinic Keratosis":              "HIGH",
    "Atopic Dermatitis":              "MEDIUM",
    "Benign Keratosis":               "LOW",
    "Dermatofibroma":                 "LOW",
    "Melanocytic Nevus":              "MEDIUM",
    "Melanoma":                       "HIGH",
    "Squamous Cell Carcinoma":        "HIGH",
    "Tinea / Ringworm / Candidiasis": "MEDIUM",
    "Vascular Lesion":                "MEDIUM",
}

# ── Same preprocessing as predict.py ─────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (standard for SqueezeNet)
        std= [0.229, 0.224, 0.225],
    ),
])

# ── Load model once at startup ────────────────────────────────────────────────
def load_model(path: str, num_classes: int = 9):
    model = models.squeezenet1_1(weights=None)
    # Replace the final classifier to match your training setup
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes

    state = torch.load(path, map_location="cpu")
    # Support both raw state-dict and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

try:
    MODEL = load_model(MODEL_PATH)
    print(f"[DermDx] Model loaded from '{MODEL_PATH}'")
except FileNotFoundError:
    MODEL = None
    print(f"[DermDx] WARNING: '{MODEL_PATH}' not found — /predict will return 503")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Guard: model must be loaded
    if MODEL is None:
        return jsonify({"error": f"Model file '{MODEL_PATH}' not found on server."}), 503

    # 2. Guard: image must be in the request
    if "image" not in request.files:
        return jsonify({"error": "No image field in request. Send as multipart/form-data with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename — please attach an image."}), 400

    # 3. Decode & preprocess
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0)   # [1, 3, 224, 224]
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {str(exc)}"}), 422

    # 4. Inference
    with torch.no_grad():
        logits = MODEL(tensor)                   # [1, 9]
        probs  = torch.softmax(logits, dim=1)[0] # [9]

    # 5. Build response
    probs_list  = probs.tolist()
    top_idx     = int(probs.argmax())
    top_class   = CLASS_NAMES[top_idx]
    confidence  = round(probs_list[top_idx] * 100, 2)

    all_probs = {
        name: round(probs_list[i] * 100, 2)
        for i, name in enumerate(CLASS_NAMES)
    }

    return jsonify({
        "prediction":        top_class,
        "confidence":        confidence,          # e.g. 94.3  (percentage)
        "risk":              RISK_LEVELS[top_class],
        "all_probabilities": all_probs,           # {class: pct, ...}
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
