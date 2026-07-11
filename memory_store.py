"""
memory_store.py
===============
Handles two of the three memory layers used by the agent.

    Layer 1 — Working memory
        Managed automatically by LangGraph's InvestigationState dict.
        No code needed here.

    Layer 2 — Session memory  (in-process dict)
        Per-user prediction history for the current server session.
        Keyed by Flask session ID (uuid4).
        Resets on server restart — sessions are ephemeral by design.

    Layer 3 — Long-term memory  (PostgreSQL)
        Every prediction is written to the predictions table.
        Enables:
          - Drift detection (compare submitted feature distributions
            to training data over time)
          - Aggregate stats fed back to the Communicator Agent
          - Full audit trail of every prediction ever made
          - SQL queries for analysis:
              SELECT AVG(rainfall), AVG(probability)
              FROM predictions
              WHERE timestamp > NOW() - INTERVAL '30 days'

Usage (called by app.py):
    from memory_store import (
        add_to_session,
        get_session_summary,
        log_prediction_to_db,
        get_aggregate_stats,
    )
"""

import os
import json
import time
import threading
from collections import defaultdict

import psycopg2
import psycopg2.extras

# ── Database connection string ────────────────────────────────────────────────
# Falls back to local defaults if DATABASE_URL is not set —
# makes local dev without Docker Compose still work (just run Postgres locally)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://aquifer_user:aquifer_pass@localhost:5432/aquifer"
)

# ── Thread lock ───────────────────────────────────────────────────────────────
# Protects the in-process session store from concurrent request corruption.
_lock = threading.Lock()

# ── Session store ─────────────────────────────────────────────────────────────
_session_store: dict[str, list] = defaultdict(list)


# ── Database helpers ──────────────────────────────────────────────────────────

def _get_conn():
    """
    Open a new psycopg2 connection.
    Each request gets its own connection — simple and safe for low-to-medium
    traffic. For high traffic, swap this for a connection pool:
        from psycopg2 import pool
        _pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    """
    return psycopg2.connect(DATABASE_URL)


# ── Session memory (Layer 2) ──────────────────────────────────────────────────

def add_to_session(session_id: str, record: dict) -> None:
    """
    Append a prediction record to this user's in-memory session history.
    Capped at 20 entries — oldest are silently dropped beyond that.

    Args:
        session_id : uuid4 string assigned by Flask
        record     : prediction result dict (features, prediction,
                     probability, place_name, lat, lon)
    """
    with _lock:
        _session_store[session_id].append({
            **record,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        _session_store[session_id] = _session_store[session_id][-20:]


def get_session_history(session_id: str) -> list:
    """Return full session history for this user (newest last)."""
    with _lock:
        return list(_session_store.get(session_id, []))


def get_session_summary(session_id: str) -> dict | None:
    """
    Build a concise summary of this session's predictions.
    Fed to the Communicator Agent so it can compare the current
    site to previous ones checked in the same session.

    Returns None if this is the first prediction (nothing to compare).

    Args:
        session_id : Flask session ID

    Returns:
        Summary dict or None.
    """
    history = get_session_history(session_id)

    if not history:
        return None

    n_likely   = sum(1 for r in history if r.get("prediction") == 1)
    n_unlikely = len(history) - n_likely
    best       = max(history, key=lambda r: r.get("probability", 0))
    worst      = min(history, key=lambda r: r.get("probability", 0))

    return {
        "total_checked":              len(history),
        "groundwater_likely_count":   n_likely,
        "groundwater_unlikely_count": n_unlikely,
        "highest_probability_site": {
            "place":       best.get("place_name", "unknown"),
            "probability": best.get("probability", 0),
        },
        "lowest_probability_site": {
            "place":       worst.get("place_name", "unknown"),
            "probability": worst.get("probability", 0),
        },
        "last_prediction": history[-1],
    }


# ── Long-term memory (Layer 3) ────────────────────────────────────────────────

def log_prediction_to_db(
    session_id:          str,
    feature_values:      dict,
    prediction:          int,
    probability:         float,
    matched_lat:         float,
    matched_lon:         float,
    place_name:          str,
    reasoning_source:    str,
    sensitivity_results: list,
    flagged_conflicts:   list,
) -> None:
    """
    Write one prediction to the PostgreSQL predictions table.
    Also upserts a row in the sessions table to track unique users.

    Args:
        session_id          : Flask session uuid
        feature_values      : dict of {FEATURE_NAME: value}
        prediction          : 0 or 1
        probability         : float 0.0 – 1.0
        matched_lat/lon     : coordinates of nearest matched survey point
        place_name          : reverse-geocoded place name
        reasoning_source    : 'huggingface' or 'template' or 'template_fallback'
        sensitivity_results : list of sensitivity check dicts from agent
        flagged_conflicts   : list of conflict description strings from agent
    """
    insert_prediction = """
        INSERT INTO predictions (
            session_id,
            elevation, curvature, drainage, lithology, lulc,
            ndvi, rainfall, slope, spi, twi,
            prediction, probability,
            matched_lat, matched_lon, place_name,
            reasoning_source, sensitivity_results, flagged_conflicts
        ) VALUES (
            %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
    """

    upsert_session = """
        INSERT INTO sessions (session_id)
        VALUES (%s)
        ON CONFLICT (session_id)
        DO UPDATE SET
            last_seen = NOW(),
            n_queries = sessions.n_queries + 1
    """

    fv = feature_values   # shorthand

    try:
        conn = _get_conn()
        cur  = conn.cursor()

        cur.execute(insert_prediction, (
            session_id,
            fv.get("ELEVATION"), fv.get("CURVATURE"), fv.get("DRAINAGE"),
            fv.get("LITHOLOGY"), fv.get("LULC"),
            fv.get("NDVI"),      fv.get("RAINFALL"),  fv.get("SLOPE"),
            fv.get("SPI"),       fv.get("TWI"),
            prediction,  probability,
            matched_lat, matched_lon, place_name,
            reasoning_source,
            json.dumps(sensitivity_results),
            flagged_conflicts,
        ))

        cur.execute(upsert_session, (session_id,))

        conn.commit()
        cur.close()
        conn.close()

    except psycopg2.Error as e:
        # Never let a DB write failure crash the prediction response.
        # Log the error and continue — the user still gets their result.
        print(f"[memory_store] DB write failed (non-fatal): {e}")


def get_aggregate_stats() -> dict:
    """
    Query the predictions table for population-level statistics.
    Fed to the Communicator Agent as long-term context.

    Also useful for drift detection — compare
    population_feature_averages against training data means
    in model/feature_stats.json to spot distribution shift.

    Returns:
        Dict of stats, or {"total_predictions": 0} if table is empty
        or database is unreachable.
    """
    query = """
        SELECT
            COUNT(*)                        AS total_predictions,
            ROUND(AVG(prediction) * 100, 1) AS groundwater_likely_pct,
            ROUND(AVG(probability)::numeric, 3) AS average_probability,
            ROUND(AVG(elevation)::numeric, 2)   AS avg_elevation,
            ROUND(AVG(curvature)::numeric, 2)   AS avg_curvature,
            ROUND(AVG(drainage)::numeric, 2)    AS avg_drainage,
            ROUND(AVG(lithology)::numeric, 2)   AS avg_lithology,
            ROUND(AVG(lulc)::numeric, 2)        AS avg_lulc,
            ROUND(AVG(ndvi)::numeric, 2)        AS avg_ndvi,
            ROUND(AVG(rainfall)::numeric, 2)    AS avg_rainfall,
            ROUND(AVG(slope)::numeric, 2)       AS avg_slope,
            ROUND(AVG(spi)::numeric, 2)         AS avg_spi,
            ROUND(AVG(twi)::numeric, 2)         AS avg_twi
        FROM predictions
    """

    try:
        conn = _get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query)
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row or row["total_predictions"] == 0:
            return {"total_predictions": 0}

        return {
            "total_predictions":      int(row["total_predictions"]),
            "groundwater_likely_pct": float(row["groundwater_likely_pct"] or 0),
            "average_probability":    float(row["average_probability"] or 0),
            "population_feature_averages": {
                "ELEVATION": float(row["avg_elevation"] or 0),
                "CURVATURE": float(row["avg_curvature"] or 0),
                "DRAINAGE":  float(row["avg_drainage"]  or 0),
                "LITHOLOGY": float(row["avg_lithology"] or 0),
                "LULC":      float(row["avg_lulc"]      or 0),
                "NDVI":      float(row["avg_ndvi"]       or 0),
                "RAINFALL":  float(row["avg_rainfall"]  or 0),
                "SLOPE":     float(row["avg_slope"]     or 0),
                "SPI":       float(row["avg_spi"]       or 0),
                "TWI":       float(row["avg_twi"]       or 0),
            },
        }

    except psycopg2.Error as e:
        # DB unreachable — return empty stats rather than crashing
        print(f"[memory_store] Could not fetch aggregate stats: {e}")
        return {"total_predictions": 0}