"""
server_models.py — Local Flask server that runs the REAL ML models.
Replaces the Gemini-based server.ts for local development.

Usage:
    pip install flask flask-cors scikit-learn joblib torch transformers tensorflow
    python server_models.py

Then run your Vite frontend normally:
    npm run dev

Your Vite app will call http://localhost:3000/api/inference exactly the same
way it always has — no changes needed in the frontend.
"""

import os, json, re, joblib, warnings
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODELS_DIR = "saved_models"

# ─────────────────────────────────────────────────────────────────────────────
# Load all 4 models at startup
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🚀 Loading models...")
print("=" * 60)

# ── Phase 1: TF-IDF + Logistic Regression ────────────────────────────────────
tfidf_word = tfidf_char = tfidf_models = None
try:
    tfidf_word   = joblib.load(f"{MODELS_DIR}/tfidf_word_vectorizer.joblib")
    tfidf_char   = joblib.load(f"{MODELS_DIR}/tfidf_char_vectorizer.joblib")
    tfidf_models = joblib.load(f"{MODELS_DIR}/tfidf_logreg_models.joblib")
    print("✅ Phase 1 (TF-IDF + LogReg) loaded")
except Exception as e:
    print(f"⚠️  Phase 1 not loaded: {e}")
    print("   → Run the TF-IDF notebook first to generate saved_models/")

# ── Phase 2: BiLSTM + BiGRU (Keras) ──────────────────────────────────────────
lstm_model = lstm_tokenizer = lstm_max_len = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    lstm_model     = load_model(f"{MODELS_DIR}/lstm_model.h5")
    lstm_tokenizer = joblib.load(f"{MODELS_DIR}/lstm_tokenizer.joblib")
    with open(f"{MODELS_DIR}/lstm_config.json") as f:
        cfg = json.load(f)
    lstm_max_len = cfg["MAX_LEN"]
    print("✅ Phase 2 (BiLSTM + BiGRU) loaded")
except Exception as e:
    print(f"⚠️  Phase 2 not loaded: {e}")
    print("   → Run the LSTM notebook first to generate saved_models/")

# ── Phase 3: DistilBERT ───────────────────────────────────────────────────────
distilbert_model = distilbert_tokenizer = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    distilbert_tokenizer = AutoTokenizer.from_pretrained(
        f"{MODELS_DIR}/distilbert_tokenizer"
    )
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6,
        problem_type="multi_label_classification",
    )
    weights_path = f"{MODELS_DIR}/distilbert_weights.pt"
    if os.path.exists(weights_path):
        distilbert_model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE)
        )
        print("✅ Phase 3 (DistilBERT) loaded with trained weights")
    else:
        print("⚠️  Phase 3: tokenizer found but no weights file — using untrained model")
        print(f"   → Expected: {weights_path}")
    distilbert_model.to(DEVICE).eval()
except Exception as e:
    print(f"⚠️  Phase 3 not loaded: {e}")
    print("   → Run the DistilBERT notebook first to generate saved_models/")

# ── Phase 4: RoBERTa ──────────────────────────────────────────────────────────
roberta_model = roberta_tokenizer = None
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    roberta_tokenizer = AutoTokenizer.from_pretrained(
        f"{MODELS_DIR}/roberta_tokenizer"
    )
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=6,
        problem_type="multi_label_classification",
    )
    weights_path = f"{MODELS_DIR}/roberta_weights.pt"
    if os.path.exists(weights_path):
        roberta_model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE)
        )
        print("✅ Phase 4 (RoBERTa) loaded with trained weights")
    else:
        print("⚠️  Phase 4: tokenizer found but no weights file — using untrained model")
        print(f"   → Expected: {weights_path}")
    roberta_model.to(DEVICE).eval()
except Exception as e:
    print(f"⚠️  Phase 4 not loaded: {e}")
    print("   → Run the RoBERTa notebook first to generate saved_models/")

print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_tfidf(text):
    """TF-IDF preprocessing (matches tf_idf_results notebook Cell 8)"""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^\w\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_transformer(text):
    """Transformer preprocessing (minimal — BERT/RoBERTa handle their own tokenization)"""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Per-model inference functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_tfidf(text):
    from scipy.sparse import hstack
    clean = preprocess_tfidf(text)
    X_word = tfidf_word.transform([clean])
    X_char = tfidf_char.transform([clean])
    X      = hstack([X_word, X_char])
    result = {}
    for label in LABELS:
        clf = tfidf_models[label]
        prob = clf.predict_proba(X)[0][1]
        result[label] = float(prob)
    return result


def predict_lstm(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    clean = preprocess_tfidf(text)          # same preprocessing as training
    seq   = lstm_tokenizer.texts_to_sequences([clean])
    pad   = pad_sequences(seq, maxlen=lstm_max_len, padding="post", truncating="post")
    probs = lstm_model.predict(pad, verbose=0)[0]
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


def predict_transformer(text, tokenizer, model):
    import torch
    DEVICE = next(model.parameters()).device
    clean  = preprocess_transformer(text)
    enc    = tokenizer(
        clean,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        )
    probs = torch.sigmoid(out.logits).cpu().numpy()[0]
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


# ─────────────────────────────────────────────────────────────────────────────
# /api/inference — same endpoint your Vite app already calls
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/inference", methods=["POST"])
def inference():
    try:
        data  = request.json
        text  = (data.get("text") or "").strip()
        model = (data.get("model") or "").strip()

        if not text or not model:
            return jsonify({"error": "text and model are required"}), 400

        # ── Route to the correct model ────────────────────────────────────────
        if model == "TF-IDF + LogReg":
            if tfidf_models is None:
                return jsonify({"error": "TF-IDF model not loaded. Run the TF-IDF notebook and generate saved_models/."}), 503
            scores = predict_tfidf(text)

        elif model == "LSTM":
            if lstm_model is None:
                return jsonify({"error": "LSTM model not loaded. Run the LSTM notebook and generate saved_models/."}), 503
            scores = predict_lstm(text)

        elif model == "DistilBERT":
            if distilbert_model is None:
                return jsonify({"error": "DistilBERT model not loaded. Run the DistilBERT notebook and generate saved_models/."}), 503
            scores = predict_transformer(text, distilbert_tokenizer, distilbert_model)

        elif model == "RoBERTa":
            if roberta_model is None:
                return jsonify({"error": "RoBERTa model not loaded. Run the RoBERTa notebook and generate saved_models/."}), 503
            scores = predict_transformer(text, roberta_tokenizer, roberta_model)

        else:
            return jsonify({"error": f"Unknown model: {model}"}), 400

        return jsonify(scores)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "TF-IDF + LogReg": tfidf_models is not None,
            "LSTM":            lstm_model is not None,
            "DistilBERT":      distilbert_model is not None,
            "RoBERTa":         roberta_model is not None,
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# Serve Vite frontend (for when you run `npm run build`)
# Optional — only needed if you want to test the full built app locally.
# For dev, just keep both servers running separately.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")
    if path and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    return send_from_directory(dist_dir, "index.html")


if __name__ == "__main__":
    print("📍 Server running at http://localhost:3000")
    print("🏥 Health: http://localhost:3000/api/health")
    print("📮 Inference: POST http://localhost:3000/api/inference")
    print("\nExpected saved_models/ structure:")
    print("  saved_models/")
    print("  ├── tfidf_word_vectorizer.joblib   ← from TF-IDF notebook")
    print("  ├── tfidf_char_vectorizer.joblib   ← from TF-IDF notebook")
    print("  ├── tfidf_logreg_models.joblib     ← from TF-IDF notebook")
    print("  ├── lstm_model.h5                  ← from LSTM notebook")
    print("  ├── lstm_tokenizer.joblib          ← from LSTM notebook")
    print("  ├── lstm_config.json               ← from LSTM notebook")
    print("  ├── distilbert_tokenizer/          ← from DistilBERT notebook")
    print("  ├── distilbert_weights.pt          ← from DistilBERT notebook")
    print("  ├── roberta_tokenizer/             ← from RoBERTa notebook")
    print("  └── roberta_weights.pt             ← from RoBERTa notebook")
    print()
    app.run(host="0.0.0.0", port=3000, debug=False)
