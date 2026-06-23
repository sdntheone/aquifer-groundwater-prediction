"""
predict.py
Runs the trained model on prediction_point.xlsx and produces:
    data/predictions.csv      -> all rows + prediction + probability
    static/points.geojson     -> GeoJSON FeatureCollection for the map UI
                                  (training points + prediction points)
"""
import json
import joblib
import pandas as pd
from pathlib import Path
from reasoning import generate_reasoning

BASE = Path(__file__).parent
DATA = BASE / "data"
MODEL_DIR = BASE / "model"
STATIC = BASE / "static"
STATIC.mkdir(exist_ok=True)

FEATURE_COLS = json.loads((MODEL_DIR / "feature_columns.json").read_text())
TARGET_COL_CANDIDATES = ["well presence", "Well Presence", "WELL_PRESENCE", "well_presence"]


def _load(path, is_train):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    rename_map = {c: c.upper() for c in df.columns if c.upper() in FEATURE_COLS}
    df = df.rename(columns=rename_map)
    target_col = next((c for c in TARGET_COL_CANDIDATES if c in df.columns), None)
    if target_col:
        df = df.rename(columns={target_col: "ACTUAL"})
    df["__source__"] = "train" if is_train else "prediction"
    return df


def main():
    model = joblib.load(MODEL_DIR / "rf_model.joblib")

    train_df = _load(DATA / "TRAIN_POINT.xlsx", is_train=True)
    pred_df = _load(DATA / "prediction_point.xlsx", is_train=False)

    all_df = pd.concat([train_df, pred_df], ignore_index=True, sort=False)

    X = all_df[FEATURE_COLS]
    all_df["PRED_PROB"] = model.predict_proba(X)[:, 1]
    all_df["PRED_LABEL"] = model.predict(X)

    out_csv = DATA / "predictions.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"Saved combined predictions -> {out_csv} ({len(all_df)} rows)")

    # Build GeoJSON for the Leaflet map. To keep payload size reasonable
    # and the map responsive in the browser, full text reasoning is
    # generated on-demand via the /api/predict endpoint, not baked into
    # every one of the (potentially thousands of) markers.
    features = []
    for _, row in all_df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["POINT_X"]), float(row["POINT_Y"])],
            },
            "properties": {
                "source": row["__source__"],
                "prediction": int(row["PRED_LABEL"]),
                "probability": round(float(row["PRED_PROB"]), 4),
                "actual": (int(row["ACTUAL"]) if "ACTUAL" in all_df.columns and pd.notna(row.get("ACTUAL")) else None),
            },
        })

    geojson_full = {"type": "FeatureCollection", "features": features}
    (STATIC / "points_full.geojson").write_text(json.dumps(geojson_full))

    # The map UI starts mostly empty by design (predictions appear only after
    # the user submits the feature form) — so we only ship a tiny sample of
    # 2 reference points here (one positive, one negative) just for visual
    # context of the study region. The full set is kept as points_full.geojson
    # in case you want an "explore all points" toggle later.
    pos_sample = all_df[all_df["PRED_LABEL"] == 1].iloc[[0]]
    neg_sample = all_df[all_df["PRED_LABEL"] == 0].iloc[[0]]
    sample_df = pd.concat([pos_sample, neg_sample])

    sample_features = []
    for _, row in sample_df.iterrows():
        sample_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["POINT_X"]), float(row["POINT_Y"])]},
            "properties": {
                "source": row["__source__"],
                "prediction": int(row["PRED_LABEL"]),
                "probability": round(float(row["PRED_PROB"]), 4),
                "sample": True,
            },
        })
    geojson_sample = {"type": "FeatureCollection", "features": sample_features}
    out_geo = STATIC / "points.geojson"
    out_geo.write_text(json.dumps(geojson_sample))
    print(f"Saved sample GeoJSON (map starting context) -> {out_geo} ({len(sample_features)} points)")
    print(f"Saved full GeoJSON -> {STATIC / 'points_full.geojson'} ({len(features)} points)")

    # quick sanity check: print one example reasoning
    sample = all_df.iloc[0]
    print("\n--- Example reasoning ---")
    print(generate_reasoning(
        {c: sample[c] for c in FEATURE_COLS},
        int(sample["PRED_LABEL"]),
        float(sample["PRED_PROB"]),
    ))


if __name__ == "__main__":
    main()
