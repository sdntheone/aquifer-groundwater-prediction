"""
init_db.py
==========
Creates all PostgreSQL tables and indexes needed by the application.

Run once before starting the app for the first time.
Safe to run multiple times — all statements use IF NOT EXISTS.

Usage:
    # With docker-compose DB running:
    python init_db.py

    # Or inside the running app container:
    docker-compose exec app python init_db.py

Tables:
    predictions
        One row per prediction request made through /api/predict_features.
        Stores all 10 feature inputs, model output, matched coordinates,
        agent metadata, and a timestamp.
        Used for:
            - Drift detection (track feature distribution over time)
            - Aggregate stats fed back to the Communicator Agent
            - Full audit trail of every prediction

    sessions
        One row per unique Flask/FastAPI session.
        Tracks how many queries each session has made.
        Used for basic usage analytics.

Schema decisions:
    - SMALLINT for feature values (they are integer classes 1-5)
    - REAL for probability (4-byte float, plenty of precision)
    - JSONB for sensitivity_results (variable-length, queryable with SQL)
    - TEXT[] for flagged_conflicts (simple array of strings)
    - No ORM — raw psycopg2 keeps the dependency footprint small
      and makes the SQL explicit and auditable
"""

import os
import sys
import psycopg2

# ── Connection ────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://aquifer_user:aquifer_pass@localhost:5432/aquifer",
)


def get_connection():
    """
    Open a psycopg2 connection with autocommit enabled.
    Prints a helpful error and exits cleanly if the DB is unreachable.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        return conn
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Could not connect to PostgreSQL.\n{e}")
        print("\nMake sure Docker Compose is running:")
        print("    docker-compose up db -d")
        sys.exit(1)


# ── DDL statements ────────────────────────────────────────────────────────────

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL          PRIMARY KEY,
    session_id      TEXT            NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- 10 feature inputs (discretized integer classes 1–5)
    elevation       SMALLINT,
    curvature       SMALLINT,
    drainage        SMALLINT,
    lithology       SMALLINT,
    lulc            SMALLINT,
    ndvi            SMALLINT,
    rainfall        SMALLINT,
    slope           SMALLINT,
    spi             SMALLINT,
    twi             SMALLINT,

    -- Model output
    prediction      SMALLINT        NOT NULL,
    probability     REAL            NOT NULL,

    -- Matched reference point
    matched_lat     DOUBLE PRECISION,
    matched_lon     DOUBLE PRECISION,
    place_name      TEXT,

    -- Agent metadata
    reasoning_source    TEXT,
    sensitivity_results JSONB,
    flagged_conflicts   TEXT[]
);
"""

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          SERIAL          PRIMARY KEY,
    session_id  TEXT            NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_seen   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    n_queries   INTEGER         NOT NULL DEFAULT 1
);
"""

# Index on timestamp — speeds up drift queries like:
#   SELECT AVG(rainfall) FROM predictions
#   WHERE timestamp > NOW() - INTERVAL '7 days'
CREATE_IDX_TIMESTAMP = """
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions (timestamp DESC);
"""

# Index on session_id — speeds up per-session lookups
CREATE_IDX_SESSION = """
CREATE INDEX IF NOT EXISTS idx_predictions_session
    ON predictions (session_id);
"""


# ── Init function ─────────────────────────────────────────────────────────────

def init_db():
    """
    Create all tables and indexes.
    Called directly when this script is run, or can be imported
    and called from a startup hook if needed.
    """
    print("Connecting to PostgreSQL...")
    conn = get_connection()
    cur  = conn.cursor()

    print("Creating predictions table...")
    cur.execute(CREATE_PREDICTIONS)

    print("Creating sessions table...")
    cur.execute(CREATE_SESSIONS)

    print("Creating indexes...")
    cur.execute(CREATE_IDX_TIMESTAMP)
    cur.execute(CREATE_IDX_SESSION)

    cur.close()
    conn.close()

    print("\nDatabase initialised successfully.")
    print("Tables created: predictions, sessions")
    print("\nNext step: python main.py")


if __name__ == "__main__":
    init_db()