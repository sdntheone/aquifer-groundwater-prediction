"""
train.py
Trains a Random Forest classifier to predict groundwater well presence
from terrain / hydrological / land-use features.

Run:
    python3 train.py
Outputs:
    model/rf_model.joblib          -> trained model
    model/feature_columns.json     -> ordered list of feature names used
    model/metrics.json             -> evaluation metrics
    model/feature_importance.json -> feature importance ranking
    model/training_log.json        -> simple "MLOps" experiment log (append-only)
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE = Path(__file__).parent
DATA = BASE / "data"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL_CANDIDATES = ["well presence", "Well Presence", "WELL_PRESENCE", "well_presence"]
COORD_COLS = ["POINT_X", "POINT_Y"]

FEATURE_COLS = [
    "ELEVATION", "CURVATURE", "DRAINAGE", "LITHOLOGY",
    "LULC", "NDVI", "RAINFALL", "SLOPE", "SPI", "TWI"
]


def load_train_data():
    df = pd.read_excel(DATA / "TRAIN_POINT.xlsx")
    df.columns = [c.strip() for c in df.columns]
    # normalize column name casing to match FEATURE_COLS
    rename_map = {c: c.upper() for c in df.columns if c.upper() in FEATURE_COLS}
    df = df.rename(columns=rename_map)
    target_col = next(c for c in TARGET_COL_CANDIDATES if c in df.columns)
    df = df.rename(columns={target_col: "TARGET"})
    return df


def main():
    print("Loading training data...")
    df = load_train_data()
    print(f"Rows: {len(df)} | Positive (well present): {df['TARGET'].sum()} "
          f"({df['TARGET'].mean()*100:.2f}%)")

    X = df[FEATURE_COLS]
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight='balanced' to handle the strong class imbalance
    # (only ~1.4% positive class in this dataset)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    # cross validation (stratified) for a more robust performance estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc_test": roc_auc_score(y_test, y_proba),
        "roc_auc_cv_mean": float(np.mean(cv_auc)),
        "roc_auc_cv_std": float(np.std(cv_auc)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, zero_division=0))

    # retrain on FULL data for the final deployed model (common practice
    # once architecture/hparams are validated via train/test split + CV)
    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    importance = dict(sorted(
        zip(FEATURE_COLS, final_model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True
    ))

    # persist artifacts
    joblib.dump(final_model, MODEL_DIR / "rf_model.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLS))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=2))

    # also persist dataset feature statistics (mean/std/min/max) used later
    # by the reasoning engine to describe whether a value is "high"/"low"
    stats = {
        col: {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "min": float(X[col].min()),
            "max": float(X[col].max()),
        } for col in FEATURE_COLS
    }
    (MODEL_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2))

    # lightweight experiment-tracking log (a poor-man's MLflow run log;
    # see README for instructions to swap in real MLflow when available)
    log_path = MODEL_DIR / "training_log.json"
    history = json.loads(log_path.read_text()) if log_path.exists() else []
    history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "RandomForestClassifier",
        "params": {
            "n_estimators": 400,
            "class_weight": "balanced",
            "min_samples_leaf": 2,
        },
        "metrics": metrics,
    })
    log_path.write_text(json.dumps(history, indent=2))

    print("\nSaved model + artifacts to", MODEL_DIR)
    print("Top features:", list(importance.items())[:5])


if __name__ == "__main__":
    main()
