# AQUIFER — Complete Setup & Usage Guide

## What is this project?

AQUIFER is a **production-ready groundwater well-presence prediction system** that combines:
- A **Random Forest ML model** trained on 2,793 survey points with terrain and climate features
- A **feature-form based interface** where users enter 10 hydrological parameters
- **LLM-powered explanations** (via Hugging Face) that clearly explain *why* a location is predicted to have groundwater
- A **live map UI** showing matched location and reverse-geocoded place names
- **Full MLOps setup**: CI/CD pipeline, Docker containerization, Render/Vercel deployment configs

---

## Quick Start (Windows PowerShell)

```powershell
# 1. Navigate into the project
cd gw_project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1

# If that fails, run once:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Train the model (optional - pre-trained model is included)
python train.py

# 7. Generate prediction dataset & map context
python predict.py

# 8. Start the app
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## How to Use the App

### 1. **Feature Glossary** (top of sidebar)
Click "What do these features mean?" to see plain-English descriptions of each input:
- **Rainfall** — How much precipitation this area receives (more = better recharge)
- **Elevation** — Height above sea level (low elevation = water collects here)
- **Slope** — How steep the terrain is (gentle slopes = water has time to soak in)
- **Land Use/Land Cover** — Forest, farmland, urban, etc. (affects infiltration rates)
- **Drainage Density** — How many streams per unit area (high density = water sheds quickly)
- **Lithology** — Rock/soil type (affects permeability and storage capacity)
- **NDVI** — Vegetation greenness (healthy plants indicate subsurface moisture)
- **Curvature** — Whether terrain curves inward (collects) or outward (sheds water)
- **TWI** — Topographic Wetness Index (model-derived terrain index predicting moisture accumulation)
- **SPI** — Stream Power Index (model-derived index of erosive flow)

### 2. **Enter Feature Values**
All 10 features have a valid range (typically 1–5, from training data). Each field shows its range as a hint. The form validates:
- Missing fields → error
- Out-of-range values → red border + suggestion to enter a valid value
- All valid → **Predict** button enabled

### 3. **Get a Prediction + Explanation**
On submit:
1. Model predicts groundwater presence (probability 0–100%)
2. Finds the closest **matching point** in the training dataset with the same feature values
3. **Reverse-geocodes** that coordinate to a place name (via OpenStreetMap)
4. Drops a **marker on the map** at the matched location
5. Shows **reasoning** explaining the prediction:
   - Which features drove the decision (ranked by importance)
   - Why that **combination** of features matters (e.g., "high rainfall + low slope + good drainage = water stays in the ground")
   - Plain-language hydrogeology reasoning

### 4. **Reasoning Source**
If you set an `HF_API_TOKEN` environment variable (see below), the reasoning comes from a **Hugging Face LLM**, which generates dynamic explanations mentioning interactions between features. Otherwise, it falls back to a **deterministic template** that still gives clear reasoning.

---

## Environment Variables (Optional)

### Hugging Face LLM Reasoning
To use an actual LLM for the explanations instead of templates:

```powershell
# In PowerShell:
$env:HF_API_TOKEN="hf_your_token_here"
$env:HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"   # optional, this is the default

python app.py
```

Or create a `.env` file in the project root:
```
HF_API_TOKEN=hf_your_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

**Getting a Hugging Face token:**
1. Go to https://huggingface.co/
2. Sign up (free) or log in
3. Settings → Access Tokens → Create New Token (read access is fine)
4. Copy and paste it into the environment variable

The app **never breaks** if the HF API is unavailable — it automatically falls back to template reasoning.

---

## Project Structure

```
gw_project/
├── app.py                          # Flask backend (API endpoints + UI)
├── train.py                        # Train the RandomForest model
├── predict.py                      # Score test points & generate map GeoJSON
├── reasoning.py                    # Deterministic explanation engine + feature info
├── llm_reasoning.py                # Hugging Face LLM call wrapper (with fallback)
├── data/
│   ├── TRAIN_POINT.xlsx           # 2,793 labeled survey points
│   ├── prediction_point.xlsx      # Points to score for the map
│   └── predictions.csv            # Generated: scores for all points
├── model/
│   ├── rf_model.joblib            # Trained RandomForest
│   ├── feature_columns.json       # List of features in order
│   ├── feature_stats.json         # Min/max/mean/std for validation + reasoning
│   ├── feature_importance.json    # Which features matter most
│   ├── metrics.json               # Training accuracy, ROC-AUC, F1, etc.
│   └── training_log.json          # Experiment history (appended on each train)
├── static/
│   ├── points.geojson             # Sample reference points for map context
│   └── points_full.geojson        # All points (for optional "explore all" feature)
├── templates/
│   └── index.html                 # Single-page Leaflet map + form UI
├── api/index.py                   # Vercel serverless entrypoint
├── requirements.txt               # Python dependencies
├── setup.sh                       # One-command setup (Linux/Mac)
├── Dockerfile                     # Containerized deployment
├── Procfile                       # Heroku/Render configuration
├── render.yaml                    # Render-specific deployment config
├── vercel.json                    # Vercel serverless config
├── .github/workflows/ml-pipeline.yml  # CI/CD: retrain + quality gate
└── README.md                      # This file
```

---

## Model Details

### Training Data
- **2,793 points** across Bundelkhand region (India)
- **10 features**: ELEVATION, CURVATURE, DRAINAGE, LITHOLOGY, LULC, NDVI, RAINFALL, SLOPE, SPI, TWI
- **Target**: binary `well presence` (1 = well confirmed, 0 = no well)
- **Class imbalance**: ~1.4% positive (39 wells) — model uses `class_weight="balanced"` to handle this

### Model Performance
- **5-fold cross-validated ROC-AUC**: ~0.98
- **Test set metrics** (20% held-out):
  - Accuracy: ~99%
  - Precision (on positive class): 100% (very few false positives)
  - Recall: ~88% (catches most true wells)
  - F1: ~0.93

### Caveats
- Only 39 positive examples → model is fragile to new data distributions
- Features are discretized classes (1–5), not continuous measurements
- "Coordinate matching" via nearest-neighbor lookup — not actual raster sampling
- Strong class imbalance means probability thresholds are learned on a tiny positive set

---

## Deployment

### Option 1: Render (Recommended)
Render handles Python/ML stacks well. Two methods:

**Method A: Native Python (render.yaml)**
1. Push to GitHub
2. In Render Dashboard: **New Web Service** → connect repo
3. Render auto-detects `render.yaml` and configures itself

**Method B: Docker**
1. Push to GitHub
2. Create service, choose **Docker** runtime
3. Render builds from `Dockerfile` automatically

### Option 2: Docker Locally
```powershell
docker build -t groundwater-app .
docker run -p 5000:5000 -e HF_API_TOKEN="hf_xxx" groundwater-app
```

### Option 3: Vercel (Light Use)
Vercel's serverless is lighter-weight but has package-size limits. Good for demos, not production. See `vercel.json` + `api/index.py`.

---

## Data Science / Model Improvement Ideas

**In priority order:**

1. **More positive examples** — the 39 wells is a tiny sample. Field-verify more wells to build a larger training set.

2. **Real raster sampling** — if you get access to the actual GIS raster layers (DEM, rainfall, NDVI, etc.), query them directly instead of nearest-neighbor lookup.

3. **SHAP explanations** — currently the LLM just gets feature importance. SHAP values give per-prediction explanations, which are more rigorous.

4. **Depth model** — add a regression model predicting water table depth (meters), not just presence/absence.

5. **Uncertainty quantification** — show prediction intervals or ensemble disagreement, not just a point probability.

6. **Model drift monitoring** — log every prediction's feature distribution and alert if it drifts from training data.

---

## Troubleshooting

**"pip install" fails on Windows with "meson" error**
→ Your Python version is newer than pandas 2.2.2 has wheels for. Update `requirements.txt` to remove the hard pin:
```
pandas>=2.2.3  # instead of pandas==2.2.2
```

**"ModuleNotFoundError: requests"**
→ You're missing the `requests` library. Run: `pip install requests`

**Map shows no points**
→ `static/points.geojson` is empty or missing. Run `python predict.py` to generate it.

**"Could not reach the prediction service"**
→ Flask app isn't running. Make sure `python app.py` is still executing in your terminal.

**Reasoning always shows "template_fallback"**
→ Either no HF token is set, or the API call failed (check internet connection, token validity).

---

## For Production Use

- [ ] Add pytest test suite for API endpoints
- [ ] Wire up MLflow for proper experiment tracking (instead of JSON log)
- [ ] Set up monitoring: log every prediction's features to detect drift
- [ ] Add a model registry / versioning system
- [ ] Collect user feedback: store predictions + actual well drilling outcomes for retraining
- [ ] Batch prediction endpoint: accept CSV, return ranked site list
- [ ] Rate limiting / authentication if exposing publicly
- [ ] Caching: predictions for the same features should be instant

---

## Citations & Data Source

This model was trained on the **Bundelkhand region of India** using publicly-available GIS layers:
- DEM: elevation derivatives (slope, curvature, aspect)
- Rainfall: gridded historical precipitation
- Land cover: satellite-derived LULC
- Vegetation: NDVI from satellite imagery
- Lithology: hydrogeological maps

Survey points with confirmed well presence are sourced from [actual data source — insert your citation here].

---

## License & Credits

AQUIFER © 2024 | Groundwater Intelligence Project

Built with:
- **scikit-learn** for ML
- **Flask** for web backend
- **Leaflet.js** for maps
- **Hugging Face Inference API** for LLM reasoning
- **OpenStreetMap / Nominatim** for reverse geocoding

For questions or improvements, open an issue or submit a PR.
