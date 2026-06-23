"""
app.py
Flask backend serving:
  GET  /                       -> map UI (templates/index.html)
  GET  /static/points.geojson  -> reference points (a small sample, see predict.py)
  GET  /api/feature_ranges     -> valid min/max per feature, for client-side form hints
  POST /api/predict_features   -> {ELEVATION, CURVATURE, ... } -> validates ranges,
                                   predicts groundwater presence, finds nearest
                                   matching coordinate in the dataset, reverse-geocodes
                                   it to a place name, and generates an LLM explanation
  GET  /api/health             -> health check
"""
import json
import requests
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from llm_reasoning import generate_llm_reasoning

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
DATA = BASE / "data"

app = Flask(__name__, static_folder="static", template_folder="templates")

# Import feature info for glossary
from reasoning import FEATURE_INFO

# ---------------------------------------------------------------------------
# Load model + reference data once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_DIR / "rf_model.joblib")
FEATURE_COLS = json.loads((MODEL_DIR / "feature_columns.json").read_text())
FEATURE_STATS = json.loads((MODEL_DIR / "feature_stats.json").read_text())

_ref = pd.read_csv(DATA / "predictions.csv")
_ref_features_raw = _ref[FEATURE_COLS].to_numpy()
_ref_coords = _ref[["POINT_X", "POINT_Y"]].to_numpy()

# Nearest-neighbor index in FEATURE space (not lat/lon) — given a user's
# feature vector, find the closest known survey/prediction point and use
# its coordinates as the best estimate of "where a place like this is".
_scaler = StandardScaler().fit(_ref_features_raw)
_nn = NearestNeighbors(n_neighbors=1).fit(_scaler.transform(_ref_features_raw))

NOMINATIM_HEADERS = {"User-Agent": "AquiferGroundwaterApp/1.0 (contact: demo@example.com)"}


def reverse_geocode(lat, lon):
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 10},
            headers=NOMINATIM_HEADERS,
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("display_name", f"{lat:.4f}, {lon:.4f}")
    except Exception:
        pass
    return f"{lat:.4f}, {lon:.4f} (place name unavailable)"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "n_reference_points": len(_ref),
        "hf_configured": bool(os.environ.get("HF_API_TOKEN", "").strip()),
    })


@app.get("/api/feature_ranges")
def feature_ranges():
    return jsonify({
        "order": FEATURE_COLS,
        "ranges": {c: {"min": FEATURE_STATS[c]["min"], "max": FEATURE_STATS[c]["max"]} for c in FEATURE_COLS},
    })


@app.get("/api/feature_info")
def feature_info():
    """Returns detailed glossary information for each feature."""
    features = {}
    for col in FEATURE_COLS:
        info = FEATURE_INFO.get(col, {})
        features[col] = {
            "label": info.get("label", col),
            "description": info.get("description", "No description available."),
        }
    return jsonify({"features": features})


@app.post("/api/predict_features")
def predict_features():
    payload = request.get_json(force=True, silent=True) or {}

    # ---- validate presence + type + range for every feature ----
    values = {}
    errors = {}
    for col in FEATURE_COLS:
        raw = payload.get(col)
        if raw is None or raw == "":
            errors[col] = "This field is required."
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors[col] = "Must be a number."
            continue
        lo, hi = FEATURE_STATS[col]["min"], FEATURE_STATS[col]["max"]
        if val < lo or val > hi:
            errors[col] = f"Out of range — enter a value between {lo:g} and {hi:g}."
            continue
        values[col] = val

    if errors:
        return jsonify({"errors": errors}), 400

    # ---- predict groundwater presence ----
    X = pd.DataFrame([values])[FEATURE_COLS]
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    # ---- find nearest matching point in FEATURE space -> its coordinates ----
    vec = _scaler.transform(X.to_numpy())
    dist, idx = _nn.kneighbors(vec, n_neighbors=1)
    idx = int(idx[0, 0])
    match_dist = float(dist[0, 0])
    lon, lat = float(_ref_coords[idx, 0]), float(_ref_coords[idx, 1])

    # ---- reverse geocode the matched coordinate to a place name ----
    place_name = reverse_geocode(lat, lon)

    # ---- generate explanation (Hugging Face LLM, falls back to template) ----
    llm_result = generate_llm_reasoning(values, pred, proba, place_name=place_name)

    return jsonify({
        "prediction": pred,
        "prediction_label": "Groundwater likely" if pred == 1 else "Groundwater unlikely",
        "probability": round(proba, 4),
        "matched_lat": lat,
        "matched_lon": lon,
        "place_name": place_name,
        "feature_space_match_distance": round(match_dist, 3),
        "reasoning": llm_result["text"],
        "reasoning_source": llm_result["source"],
        "reasoning_model": llm_result.get("model"),
        "features_used": values,
        "note": (
            "The coordinate shown is the closest matching point in the training/"
            "prediction dataset for the feature values you entered — not a "
            "reverse-engineered exact location, since many real places can share "
            "the same discretized feature combination."
        ),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
