"""
memory_store.py
===============
Manages two of the three memory layers used by the LangGraph agent.

The three layers:

    Layer 1 — Working memory
        The InvestigationState TypedDict in agent.py.
        LangGraph manages this automatically — no code here.

    Layer 2 — Session memory  (this file)
        An in-process Python dict keyed by session_id (a uuid4 string
        assigned by FastAPI's SessionMiddleware on first request).
        Stores the last 20 predictions per session.
        Resets when the server restarts — sessions are ephemeral by nature.
        For persistence across restarts, swap _session_store for Redis.

    Layer 3 — Long-term memory  (this file)
        Every prediction is written to the PostgreSQL predictions table
        via log_prediction_to_db().
        Enables:
            - Drift detection: compare submitted feature distributions
              to training data distributions over time
            - Aggregate stats fed back to the Communicator Agent as context
            - Full audit trail of every prediction ever made

Public API (imported by main.py):
    add_to_session(session_id, record)
    get_session_summary(session_id) -> dict | None
    log_prediction_to_db(session_id, feature_values, ...)
    get_aggregate_stats() -> dict

Note on function naming:
    log_prediction_to_db (not log_prediction) — the explicit suffix
    makes it obvious this is a DB write, not an in-memory operation.
    This matters when reading main.py — you immediately know what
    storage layer is being touched.
"""

import os
import json
import time
import threading
from collections import defaultdict

import psycopg2
import psycopg2.extras

# ── Database connection ───────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://aquifer_user:aquifer_pass@localhost:5432/aquifer",
)

# ── Thread safety ─────────────────────────────────────────────────────────────
# FastAPI can handle concurrent requests (especially with async endpoints).
# The in-process session store is shared across all requests in the same
# process, so reads and writes need to be protected.
_lock = threading.Lock()

# ── In-process session store ──────────────────────────────────────────────────
# defaultdict(list) means _session_store[new_key] auto-initialises to []
# instead of raising KeyError — cleaner than checking existence everywhere.
_session_store: dict[str, list] = defaultdict(list)


# ── Database helper ───────────────────────────────────────────────────────────

def _get_conn():
    """
    Open and return a new psycopg2 connection.

    One connection per request is fine for this traffic level.
    For high throughput, replace with a connection pool:
        from psycopg2 import pool
        _pool = pool.ThreadedConnectionPool(2, 20, DATABASE_URL)
        conn = _pool.getconn()
        # ... use conn ...
        _pool.putconn(conn)
    """
    return psycopg2.connect(DATABASE_URL)


# ── Layer 2: Session memory ───────────────────────────────────────────────────

def add_to_session(session_id: str, record: dict) -> None:
    """
    Append one prediction record to the user's in-memory session history.
    Silently drops the oldest entry when the cap of 20 is exceeded.

    Called by main.py after every successful prediction.

    Args:
        session_id : uuid4 string from FastAPI SessionMiddleware
        record     : dict containing at minimum prediction, probability,
                     place_name, lat, lon, features
    """
    with _lock:
        _session_store[session_id].append({
            **record,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        # Cap at 20 — slice from the right to keep newest
        _session_store[session_id] = _session_store[session_id][-20:]


def get_session_history(session_id: str) -> list:
    """
    Return the full prediction history for this session (oldest first).

    Args:
        session_id : FastAPI session ID

    Returns:
        List of prediction record dicts, or empty list.
    """
    with _lock:
        return list(_session_store.get(session_id, []))


def get_session_summary(session_id: str) -> dict | None:
    """
    Build a concise summary of this session's predictions.
    Passed to the Communicator Agent so it can compare the current
    site against previous ones checked in the same session.

    Returns None on the first prediction of a session — there's
    nothing to compare against yet, and None signals this clearly
    to both the agent and the LLM prompt builder.

    Args:
        session_id : FastAPI session ID

    Returns:
        Summary dict, or None if no history exists yet.
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
        # Give the agent the most recent prediction for direct comparison
        "last_prediction": history[-1],
    }


# ── Layer 3: Long-term memory (PostgreSQL) ────────────────────────────────────

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
    Also upserts a row in sessions to track unique session activity.

    This is a non-fatal operation — if the DB write fails (e.g. DB is
    down, connection refused), the error is printed to server logs but
    the user still receives their prediction result.

    Called by main.py after every successful agent run.

    Args:
        session_id          : FastAPI session uuid
        feature_values      : dict of {FEATURE_COL: value}
        prediction          : 0 or 1
        probability         : float 0.0–1.0
        matched_lat/lon     : coordinates of nearest feature-space match
        place_name          : reverse-geocoded location string
        reasoning_source    : 'huggingface', 'template', or 'template_fallback'
        sensitivity_results : list of sensitivity check dicts from agent
        flagged_conflicts   : list of conflict description strings from agent
    """
    sql_prediction = """
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

    # UPSERT — insert new session row, or bump n_queries if it exists
    sql_session = """
        INSERT INTO sessions (session_id)
        VALUES (%s)
        ON CONFLICT (session_id)
        DO UPDATE SET
            last_seen = NOW(),
            n_queries = sessions.n_queries + 1
    """

    fv = feature_values

    try:
        conn = _get_conn()
        cur  = conn.cursor()

        cur.execute(sql_prediction, (
            session_id,
            fv.get("ELEVATION"), fv.get("CURVATURE"), fv.get("DRAINAGE"),
            fv.get("LITHOLOGY"), fv.get("LULC"),
            fv.get("NDVI"),      fv.get("RAINFALL"),  fv.get("SLOPE"),
            fv.get("SPI"),       fv.get("TWI"),
            prediction,   probability,
            matched_lat,  matched_lon, place_name,
            reasoning_source,
            json.dumps(sensitivity_results),
            flagged_conflicts,
        ))

        cur.execute(sql_session, (session_id,))

        conn.commit()
        cur.close()
        conn.close()

    except psycopg2.Error as e:
        # Non-fatal — log and continue
        print(f"[memory_store] DB write failed (non-fatal): {e}")


def get_aggregate_stats() -> dict:
    """
    Query the predictions table for population-level statistics.

    Used by the Communicator Agent as long-term context — e.g.
    'across 150 predictions in this system, 23% were groundwater-likely.'

    Also the foundation for drift detection: compare
    population_feature_averages against model/feature_stats.json
    means to spot if submitted values are drifting from training data.

    Returns:
        Populated stats dict, or {"total_predictions": 0} if the table
        is empty or the database is unreachable.
    """
    sql = """
        SELECT
            COUNT(*)                             AS total_predictions,
            ROUND(AVG(prediction) * 100, 1)      AS groundwater_likely_pct,
            ROUND(AVG(probability)::numeric, 3)  AS average_probability,
            ROUND(AVG(elevation)::numeric,  2)   AS avg_elevation,
            ROUND(AVG(curvature)::numeric,  2)   AS avg_curvature,
            ROUND(AVG(drainage)::numeric,   2)   AS avg_drainage,
            ROUND(AVG(lithology)::numeric,  2)   AS avg_lithology,
            ROUND(AVG(lulc)::numeric,       2)   AS avg_lulc,
            ROUND(AVG(ndvi)::numeric,       2)   AS avg_ndvi,
            ROUND(AVG(rainfall)::numeric,   2)   AS avg_rainfall,
            ROUND(AVG(slope)::numeric,      2)   AS avg_slope,
            ROUND(AVG(spi)::numeric,        2)   AS avg_spi,
            ROUND(AVG(twi)::numeric,        2)   AS avg_twi
        FROM predictions
    """

    try:
        conn = _get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row or row["total_predictions"] == 0:
            return {"total_predictions": 0}

        return {
            "total_predictions":       int(row["total_predictions"]),
            "groundwater_likely_pct":  float(row["groundwater_likely_pct"] or 0),
            "average_probability":     float(row["average_probability"] or 0),
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
        print(f"[memory_store] Could not fetch aggregate stats: {e}")
        return {"total_predictions": 0}