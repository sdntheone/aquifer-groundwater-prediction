"""
predict.py
==========
Batch scoring pipeline — runs the trained model on all points in both
the training and prediction datasets, then writes two outputs:

    data/predictions.csv
        Full scored dataset (all ~3,600 points) used by main.py at startup
        to build the nearest-neighbour feature-space index. This is how
        the app maps a user's feature inputs to a real-world coordinate.

    static/points_full.geojson
        All points as GeoJSON — available for an optional "show all points"
        toggle in the UI if you add one later.

    static/points.geojson
        Two sample points only (one positive, one negative) — loaded on
        the initial map so the page isn't blank, but not overwhelming.
        The map fills in with the real matched point only after prediction.

Run after train.py:
    python predict.py
"""

import json
import joblib
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA      = BASE / "data"
MODEL_DIR = BASE / "model"
STATIC    = BASE / "static"
STATIC.mkdir(exist_ok=True)

FEATURE_COLS = json.loads(
    (MODEL_DIR / "feature_columns.json").read_text()
)

# All possible spellings of the target column in the raw Excel files
TARGET_CANDIDATES = [
    "well presence", "Well Presence", "WELL_PRESENCE", "well_presence"
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sheet(path: Path, is_train: bool) -> pd.DataFrame:
    """
    Read one Excel file, normalise column names, rename the target column.
    Adds a __source__ column so we can tell train vs prediction points apart.

    Args:
        path     : path to the .xlsx file
        is_train : True for TRAIN_POINT.xlsx, False for prediction_point.xlsx

    Returns:
        DataFrame with FEATURE_COLS + optional ACTUAL + __source__ columns
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Uppercase any column whose uppercased name matches a feature
    df = df.rename(columns={
        c: c.upper()
        for c in df.columns
        if c.upper() in FEATURE_COLS
    })

    # Rename target column to ACTUAL if present
    target = next(
        (c for c in TARGET_CANDIDATES if c in df.columns), None
    )
    if target:
        df = df.rename(columns={target: "ACTUAL"})

    df["__source__"] = "train" if is_train else "prediction"
    return df


def row_to_feature(row: pd.Series) -> dict:
    """Build a GeoJSON feature dict from one row."""
    props = {
        "source":      row["__source__"],
        "prediction":  int(row["PRED_LABEL"]),
        "probability": round(float(row["PRED_PROB"]), 4),
    }
    # Include actual label for training points (useful for accuracy overlays)
    if "ACTUAL" in row.index and pd.notna(row["ACTUAL"]):
        props["actual"] = int(row["ACTUAL"])

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(row["POINT_X"]), float(row["POINT_Y"])],
        },
        "properties": props,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("AQUIFER — Batch Prediction Pipeline")
    print("=" * 55)

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[1/4] Loading trained model...")
    model = joblib.load(MODEL_DIR / "rf_model.joblib")

    # ── Load and combine datasets ─────────────────────────────────────────
    print("\n[2/4] Loading datasets...")
    train = load_sheet(DATA / "TRAIN_POINT.xlsx",      is_train=True)
    pred  = load_sheet(DATA / "prediction_point.xlsx", is_train=False)
    all_df = pd.concat([train, pred], ignore_index=True, sort=False)
    print(f"      Combined: {len(all_df)} rows "
          f"({len(train)} train + {len(pred)} prediction)")

    # ── Score all rows ────────────────────────────────────────────────────
    print("\n[3/4] Scoring all points...")
    X = all_df[FEATURE_COLS]
    all_df["PRED_PROB"]  = model.predict_proba(X)[:, 1]
    all_df["PRED_LABEL"] = model.predict(X)

    # Save full CSV — used by main.py nearest-neighbour index at startup
    csv_path = DATA / "predictions.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"      Saved {len(all_df)} rows → {csv_path}")

    # ── Build GeoJSON files ───────────────────────────────────────────────
    print("\n[4/4] Building GeoJSON files...")

    # Full GeoJSON — all points
    all_features = [row_to_feature(row) for _, row in all_df.iterrows()]
    full_geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }
    full_path = STATIC / "points_full.geojson"
    full_path.write_text(json.dumps(full_geojson))
    print(f"      Full GeoJSON  → {full_path} ({len(all_features)} points)")

    # Sample GeoJSON — 1 positive + 1 negative point only
    # Just enough context to orient the user on the map at first load
    pos_sample = all_df[all_df["PRED_LABEL"] == 1].iloc[[0]]
    neg_sample = all_df[all_df["PRED_LABEL"] == 0].iloc[[0]]
    sample_df  = pd.concat([pos_sample, neg_sample])
    sample_features = [row_to_feature(row) for _, row in sample_df.iterrows()]
    sample_geojson  = {
        "type": "FeatureCollection",
        "features": sample_features,
    }
    sample_path = STATIC / "points.geojson"
    sample_path.write_text(json.dumps(sample_geojson))
    print(f"      Sample GeoJSON → {sample_path} ({len(sample_features)} points)")

    print("\n" + "=" * 55)
    print("Batch prediction complete.")
    n_likely = int(all_df["PRED_LABEL"].sum())
    print(f"Groundwater likely: {n_likely} / {len(all_df)} "
          f"({n_likely/len(all_df)*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()