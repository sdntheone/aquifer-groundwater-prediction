"""
main.py
=======
FastAPI backend for the AQUIFER groundwater prediction system.

Why FastAPI over Flask:
    - Pydantic models handle request validation automatically
    - Async endpoints — no blocking I/O
    - Auto-generated OpenAPI docs at /docs (try it after starting the app)
    - Type hints throughout — easier to read and maintain
    - About half the boilerplate of Flask for the same functionality

Endpoints:
    GET  /                      → 3-page map UI (templates/index.html)
    GET  /docs                  → Auto-generated API documentation (FastAPI built-in)
    GET  /api/health            → Health check (used by Render uptime monitoring)
    GET  /api/feature_ranges    → Min/max per feature for client-side form hints
    GET  /api/feature_info      → Feature descriptions for the glossary on Page 1
    POST /api/predict_features  → Main prediction endpoint — runs the LangGraph agent

Startup sequence:
    1. python train.py          (trains model + SHAP explainer)
    2. python predict.py        (scores all points + builds GeoJSON)
    3. docker-compose up db -d  (starts PostgreSQL)
    4. python init_db.py        (creates tables)
    5. python main.py           (starts the app)

Or with Docker Compose:
    docker-compose up --build
"""

import os
import uuid
import json

import joblib
import numpy as np
import pandas as pd
import requests as http_requests
import uvicorn
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, model_validator

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from reasoning import FEATURE_INFO
from agent import run_investigation
from memory_store import (
    add_to_session,
    get_session_summary,
    log_prediction_to_db,
    get_aggregate_stats,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
MODEL_DIR = BASE / "model"
DATA      = BASE / "data"

# ── Config from environment variables ─────────────────────────────────────────
SECRET_KEY   = os.environ.get("SECRET_KEY", "aquifer-dev-secret-change-in-prod")
PORT         = int(os.environ.get("PORT", 5000))

# ── FastAPI app setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="AQUIFER — Groundwater Intelligence API",
    description=(
        "Predicts groundwater well presence from terrain and climate features "
        "using a multi-agent LangGraph system with SHAP explainability."
    ),
    version="2.0.0",
)

# Session middleware — signs cookies with SECRET_KEY
# Must be added before mounting static files
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Serve static files (GeoJSON, etc.) at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template rendering
templates = Jinja2Templates(directory="templates")

# ── Load model artifacts at startup ───────────────────────────────────────────
# These are loaded once when the module is imported — not on every request.
# If the files don't exist, the app fails loudly at startup (correct behaviour
# — you must run train.py and predict.py first).
FEATURE_COLS  = json.loads((MODEL_DIR / "feature_columns.json").read_text())
FEATURE_STATS = json.loads((MODEL_DIR / "feature_stats.json").read_text())

# Reference dataset — used to find the nearest matching coordinate
# for a given set of feature values
_ref          = pd.read_csv(DATA / "predictions.csv")
_ref_features = _ref[FEATURE_COLS].to_numpy()
_ref_coords   = _ref[["POINT_X", "POINT_Y"]].to_numpy()

# StandardScaler + NearestNeighbors in FEATURE SPACE (not lat/lon)
# This finds the closest known survey point with similar terrain/climate
# characteristics to what the user entered — not the closest by distance
_scaler = StandardScaler().fit(_ref_features)
_nn     = NearestNeighbors(n_neighbors=1).fit(
    _scaler.transform(_ref_features)
)

NOMINATIM_HEADERS = {
    "User-Agent": "AquiferGroundwaterApp/2.0 (groundwater-prediction)"
}


# ── Pydantic request model ────────────────────────────────────────────────────

class FeatureInput(BaseModel):
    """
    Pydantic model for the 10 feature inputs.

    FastAPI automatically:
        - Parses the JSON request body into this model
        - Returns HTTP 422 if any field is missing or not a number
        - Shows this model in /docs with example values

    Range validation (against FEATURE_STATS min/max) is done separately
    in the endpoint because FEATURE_STATS is loaded at runtime, not at
    class definition time.
    """
    ELEVATION: float
    CURVATURE: float
    DRAINAGE:  float
    LITHOLOGY: float
    LULC:      float
    NDVI:      float
    RAINFALL:  float
    SLOPE:     float
    SPI:       float
    TWI:       float

    model_config = {
    "json_schema_extra": {
        "example": {
            "ELEVATION": 2,
            "CURVATURE": 2,
            "DRAINAGE":  4,
            "LITHOLOGY": 3,
            "LULC":      3,
            "NDVI":      3,
            "RAINFALL":  3,
            "SLOPE":     2,
            "SPI":       3,
            "TWI":       4,
        }
    }
}


# ── Custom 422 handler ────────────────────────────────────────────────────────
# Converts Pydantic's default 422 error format into our {"errors": {field: msg}}
# format so the frontend only needs to handle one error shape.

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = {}
    for error in exc.errors():
        # error['loc'] is a tuple like ('body', 'ELEVATION')
        field = error["loc"][-1] if error["loc"] else "unknown"
        errors[str(field)] = error["msg"]
    return JSONResponse(status_code=400, content={"errors": errors})


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_ranges(values: dict) -> dict:
    """
    Check each feature value is within the training data range.
    Returns a dict of {field: error_message} — empty dict means all valid.

    Called inside the prediction endpoint after Pydantic has already
    confirmed that all values are valid floats.

    Args:
        values : dict of {FEATURE_COL: float}

    Returns:
        dict of {FEATURE_COL: error_string} — empty if all valid
    """
    errors = {}
    for col, val in values.items():
        lo = FEATURE_STATS[col]["min"]
        hi = FEATURE_STATS[col]["max"]
        if val < lo or val > hi:
            errors[col] = f"Enter a value between {lo:g} and {hi:g}."
    return errors


def find_nearest_coordinate(values: dict) -> tuple[float, float, float]:
    """
    Find the nearest point in the reference dataset in FEATURE SPACE.
    Returns (lat, lon, distance) where distance is the scaled Euclidean
    distance in feature space (not geographic distance).

    This is how the app maps user feature inputs to a real-world location —
    we find the closest known survey point with matching terrain characteristics.

    Args:
        values : validated feature dict

    Returns:
        (lat, lon, feature_space_distance)
    """
    X    = pd.DataFrame([values])[FEATURE_COLS]
    vec  = _scaler.transform(X.to_numpy())
    dist, idx = _nn.kneighbors(vec)
    idx  = int(idx[0, 0])
    lon  = float(_ref_coords[idx, 0])
    lat  = float(_ref_coords[idx, 1])
    return lat, lon, float(dist[0, 0])


def reverse_geocode(lat: float, lon: float) -> str:
    """
    Convert lat/lon to a human-readable place name using
    OpenStreetMap's Nominatim API (free, no key required).

    Returns the coordinate string if the API is unreachable
    — app never fails because of a geocoding error.

    Args:
        lat, lon : coordinate of the matched reference point

    Returns:
        Place name string or fallback coordinate string
    """
    try:
        resp = http_requests.get(
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
    return f"{lat:.4f}, {lon:.4f}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the 3-page map UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/api/health")
async def health():
    """
    Health check endpoint.
    Render and uptime monitors hit this to confirm the app is alive.
    Returns key config flags so you can verify env vars are loaded.
    """
    return {
        "status":                 "ok",
        "n_reference_points":     len(_ref),
        "hf_configured":          bool(os.environ.get("HF_API_TOKEN", "").strip()),
        "langsmith_configured":   bool(os.environ.get("LANGCHAIN_API_KEY", "").strip()),
        "db_url_configured":      bool(os.environ.get("DATABASE_URL", "").strip()),
    }


@app.get("/api/feature_ranges")
async def feature_ranges():
    """
    Return the valid min/max range for each feature.
    Used by the frontend to:
        - Show range hints in each form field
        - Validate values client-side before submitting
    """
    return {
        "order":  FEATURE_COLS,
        "ranges": {
            col: {
                "min": FEATURE_STATS[col]["min"],
                "max": FEATURE_STATS[col]["max"],
            }
            for col in FEATURE_COLS
        },
    }


@app.get("/api/feature_info")
async def feature_info():
    """
    Return plain-English descriptions and metadata for each feature.
    Used by the frontend to:
        - Render the glossary section on Page 1
        - Show descriptions under each form field on Page 2
        - Label SHAP and sensitivity charts on Page 3
    """
    return {
        "features": {
            col: {
                "label":       FEATURE_INFO.get(col, {}).get("label", col),
                "description": FEATURE_INFO.get(col, {}).get("description", ""),
                "favorable":   FEATURE_INFO.get(col, {}).get("favorable", "context"),
            }
            for col in FEATURE_COLS
        }
    }


@app.post("/api/predict_features")
async def predict_features(request: Request, body: FeatureInput):
    """
    Main prediction endpoint — runs the full LangGraph multi-agent investigation.

    Flow:
        1. Pydantic validates types (auto — handled by FastAPI)
        2. Range validation against FEATURE_STATS
        3. Nearest-neighbour lookup in feature space → coordinate
        4. Reverse geocode coordinate → place name
        5. Retrieve session memory + long-term aggregate stats
        6. Run LangGraph: Investigator Agent → Communicator Agent
        7. Log prediction to PostgreSQL
        8. Update session memory
        9. Return full result JSON

    Args:
        request : FastAPI Request (needed for session access)
        body    : FeatureInput Pydantic model (auto-validated)

    Returns:
        JSON with prediction, probability, SHAP values, reasoning,
        agent trace, matched location, and memory context
    """
    # ── Range validation ──────────────────────────────────────────────────
    values = body.model_dump()
    errors = validate_ranges(values)
    if errors:
        return JSONResponse(status_code=400, content={"errors": errors})

    # ── Session management ────────────────────────────────────────────────
    # FastAPI sessions work identically to Flask sessions here
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    sid = request.session["session_id"]

    # ── Retrieve memory context ───────────────────────────────────────────
    session_summary = get_session_summary(sid)  # Layer 2 memory
    agg_stats       = get_aggregate_stats()      # Layer 3 memory

    # ── Find nearest matching coordinate ─────────────────────────────────
    lat, lon, match_dist = find_nearest_coordinate(values)
    place_name           = reverse_geocode(lat, lon)

    # ── Run multi-agent investigation ─────────────────────────────────────
    # This is where the LangGraph graph is invoked.
    # Investigator Agent runs tools, then hands off to Communicator Agent.
    # LangSmith traces this automatically if LANGCHAIN_TRACING_V2=true.
    result = run_investigation(
        feature_values  = values,
        session_id      = sid,
        session_summary = session_summary,
        agg_stats       = agg_stats,
        place_name      = place_name,
    )

    # ── Log to PostgreSQL (Layer 3 long-term memory) ──────────────────────
    log_prediction_to_db(
        session_id          = sid,
        feature_values      = values,
        prediction          = result["prediction"],
        probability         = result["probability"],
        matched_lat         = lat,
        matched_lon         = lon,
        place_name          = place_name,
        reasoning_source    = result["reasoning_source"],
        sensitivity_results = result["sensitivity_results"],
        flagged_conflicts   = result["flagged_conflicts"],
    )

    # ── Update session memory (Layer 2) ───────────────────────────────────
    add_to_session(sid, {
        "features":    values,
        "prediction":  result["prediction"],
        "probability": result["probability"],
        "place_name":  place_name,
        "lat": lat, "lon": lon,
    })

    # ── Return response ───────────────────────────────────────────────────
    return {
        "prediction":            result["prediction"],
        "prediction_label":      (
            "Groundwater likely"
            if result["prediction"] == 1
            else "Groundwater unlikely"
        ),
        "probability":           round(result["probability"], 4),
        "matched_lat":           lat,
        "matched_lon":           lon,
        "place_name":            place_name,
        "feature_match_distance":round(match_dist, 3),
        "shap_values":           result["shap_values"],
        "sensitivity_results":   result["sensitivity_results"],
        "flagged_conflicts":     result["flagged_conflicts"],
        "investigation_notes":   result["investigation_notes"],
        "investigation_report":  result["investigation_report"],
        "reasoning":             result["final_reasoning"],
        "reasoning_source":      result["reasoning_source"],
        "reasoning_model":       result["reasoning_model"],
        "session_summary":       session_summary,
        "aggregate_stats":       agg_stats,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )