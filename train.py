"""
train.py
========
Trains a RandomForest classifier to predict groundwater well presence
from 10 terrain, hydrology, and land-use features.

Run this first before anything else:
    python train.py

Artifacts written to model/:
    rf_model.joblib          - trained RandomForest (full dataset)
    shap_explainer.joblib    - TreeExplainer for per-prediction SHAP values
    feature_columns.json     - ordered list of feature names used in training
    feature_stats.json       - min/max/mean/std per feature
                               used by main.py for form validation
                               used by reasoning.py for _level() classification
    feature_importance.json  - global feature importance ranking (descending)
    metrics.json             - evaluation metrics: accuracy, ROC-AUC, F1, etc.
    training_log.json        - append-only experiment log entry per run

Key design decisions:
    class_weight='balanced'
        Without this, the model learns to always predict 0 and still gets
        98% accuracy due to the 1:70 class imbalance. Balanced weights
        force it to actually learn the minority class (wells present).

    Retrain on full dataset after evaluation
        We evaluate on a held-out split to get honest metrics, then retrain
        on ALL data before saving. The deployed model benefits from every
        labeled example, not just 80% of them.

    normalise_shap() exported as a standalone function
        agent.py imports and reuses this exact function when computing SHAP
        at prediction time — same logic, no duplication.

MLOps note:
    training_log.json is a lightweight substitute for MLflow.
    To swap in real MLflow, replace that block with:
        import mlflow
        with mlflow.start_run():
            mlflow.log_params({...})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_model, 'model')
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA      = BASE / "data"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_CANDIDATES = [
    "well presence", "Well Presence",
    "WELL_PRESENCE", "well_presence",
]

FEATURE_COLS = [
    "ELEVATION", "CURVATURE", "DRAINAGE", "LITHOLOGY",
    "LULC", "NDVI", "RAINFALL", "SLOPE", "SPI", "TWI",
]


# ── Public helper — imported by agent.py ──────────────────────────────────────

def normalise_shap(raw, n_samples: int, n_features: int) -> np.ndarray:
    """
    Normalise SHAP output to a consistent shape of (n_samples, n_features)
    for the positive class (class 1 = groundwater present).

    Different versions of the shap library and different model types return
    arrays in different shapes. This function handles every known variant so
    the rest of the codebase never has to think about it.

    Observed shapes:
        list of two arrays, each (n_samples, n_features)
            Classic binary classifier output from older shap versions.
            We take index [1] (positive class) directly.

        single 3D array (n_features, n_samples, n_classes)
            Newer shap versions sometimes return this shape.
            We slice class 1 and transpose.

        single 2D array (n_features, n_samples)
            Some versions return features as rows.
            We just transpose.

        single 2D array (n_samples, n_features)
            Already correct — return as-is.

    Args:
        raw       : raw output from explainer.shap_values()
        n_samples : number of rows in the input DataFrame
        n_features: number of feature columns (len(FEATURE_COLS))

    Returns:
        np.ndarray of shape (n_samples, n_features)

    Raises:
        ValueError if the shape cannot be resolved to (n_samples, n_features)
    """
    # Case 1: list of two arrays [class_0, class_1]
    if isinstance(raw, list) and len(raw) == 2:
        arr = np.array(raw[1])
    else:
        arr = np.array(raw)

    # Case 2: 3D array — take class 1 slice
    if arr.ndim == 3:
        # Could be (n_samples, n_features, n_classes)
        # or (n_features, n_samples, n_classes)
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        else:
            arr = arr[:, :, 0]

    # Case 3: 2D — ensure (n_samples, n_features)
    if arr.ndim == 2:
        if arr.shape == (n_features, n_samples) and n_features != n_samples:
            arr = arr.T
        elif arr.shape[0] == n_features and arr.shape[1] == n_samples:
            arr = arr.T

    if arr.shape != (n_samples, n_features):
        raise ValueError(
            f"Could not normalise SHAP array to ({n_samples}, {n_features}). "
            f"Got shape {arr.shape} from raw type {type(raw).__name__}."
        )

    return arr


# ── Data loading ──────────────────────────────────────────────────────────────

def load_train_data() -> pd.DataFrame:
    """
    Read TRAIN_POINT.xlsx, normalise column names, rename target to TARGET.

    Returns:
        DataFrame with FEATURE_COLS + TARGET column
    """
    df = pd.read_excel(DATA / "TRAIN_POINT.xlsx")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        c: c.upper()
        for c in df.columns
        if c.upper() in FEATURE_COLS
    })
    target_col = next(c for c in TARGET_CANDIDATES if c in df.columns)
    return df.rename(columns={target_col: "TARGET"})


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("AQUIFER — Model Training Pipeline")
    print("=" * 55)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading training data...")
    df   = load_train_data()
    X, y = df[FEATURE_COLS], df["TARGET"]
    pos  = int(y.sum())
    print(f"      Rows: {len(df)} | Positive: {pos} ({y.mean()*100:.2f}%)")
    print(f"      Class imbalance ratio: 1 : {int((1 - y.mean()) / y.mean())}")

    # ── 2. Split ──────────────────────────────────────────────────────────
    print("\n[2/6] Splitting data (80 / 20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Train ──────────────────────────────────────────────────────────
    print("\n[3/6] Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    print("\n[4/6] Evaluating model...")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "f1_score":         f1_score(y_test, y_pred, zero_division=0),
        "roc_auc_test":     roc_auc_score(y_test, y_proba),
        "roc_auc_cv_mean":  float(np.mean(cv_auc)),
        "roc_auc_cv_std":   float(np.std(cv_auc)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train":          int(len(X_train)),
        "n_test":           int(len(X_test)),
        "positive_rate":    float(y.mean()),
    }
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"      CV ROC-AUC: {metrics['roc_auc_cv_mean']:.4f} "
          f"(± {metrics['roc_auc_cv_std']:.4f})")

    # ── 5. Retrain on full dataset ────────────────────────────────────────
    print("\n[5/6] Retraining on full dataset for deployment...")
    final_model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    # ── 6. SHAP explainer ─────────────────────────────────────────────────
    print("\n[6/6] Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(final_model)

    # Verify on 5 samples using normalise_shap
    raw_sample = explainer.shap_values(X.iloc[:5])
    verified   = normalise_shap(raw_sample, n_samples=5, n_features=len(FEATURE_COLS))
    print(f"      Raw SHAP type   : {type(raw_sample).__name__}, "
          f"shape: {np.array(raw_sample).shape}")
    print(f"      Normalised shape: {verified.shape} "
          f"(expected (5, {len(FEATURE_COLS)})) ✓")

    # ── Save artifacts ────────────────────────────────────────────────────
    importance = dict(sorted(
        zip(FEATURE_COLS, final_model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    ))

    stats = {
        col: {
            "mean": float(X[col].mean()),
            "std":  float(X[col].std()),
            "min":  float(X[col].min()),
            "max":  float(X[col].max()),
        }
        for col in FEATURE_COLS
    }

    joblib.dump(final_model, MODEL_DIR / "rf_model.joblib")
    joblib.dump(explainer,   MODEL_DIR / "shap_explainer.joblib")

    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLS))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    (MODEL_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2))

    # Append-only experiment log
    log_path = MODEL_DIR / "training_log.json"
    history  = json.loads(log_path.read_text()) if log_path.exists() else []
    history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":     "RandomForestClassifier",
        "params": {
            "n_estimators":     400,
            "class_weight":     "balanced",
            "min_samples_leaf": 2,
        },
        "metrics": metrics,
    })
    log_path.write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 55)
    print("Training complete. Artifacts saved to model/")
    print(f"Top 5 features: {list(importance.items())[:5]}")
    print("=" * 55)


if __name__ == "__main__":
    main()