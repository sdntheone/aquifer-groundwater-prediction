"""
init_db.py
==========
Creates the PostgreSQL tables needed by the application.
Run this once before starting the app for the first time,
or whenever you want to reset the database schema.

Usage:
    # With Docker Compose running:
    python init_db.py

    # Or inside the app container:
    docker-compose exec app python init_db.py

Tables created:
    predictions   — one row per prediction request
                    stores all 10 feature values, model output,
                    matched location, and metadata
                    used for drift detection and aggregate stats

    sessions      — lightweight log of session activity
                    used to count unique users over time

Schema design notes:
    - feature values stored as SMALLINT (they are 1-5 integer classes)
    - probability stored as REAL (4-byte float, sufficient precision)
    - jsonb used for sensitivity_results (variable length, queryable)
    - no ORM used intentionally — raw psycopg2 keeps the dependency
      footprint small and makes the SQL explicit and auditable
"""

import os
import sys
import psycopg2
from psycopg2 import sql

# ── Connection ────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://aquifer_user:aquifer_pass@localhost:5432/aquifer"
)


def get_connection():
    """
    Open and return a psycopg2 connection.
    Raises a clear error if the database is not reachable.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        return conn
    except psycopg2.OperationalError as e:
        print(f"\nCould not connect to PostgreSQL: {e}")
        print("Make sure Docker Compose is running:  docker-compose up db")
        sys.exit(1)


# ── Table definitions ─────────────────────────────────────────────────────────
CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 10 feature inputs (discretized classes 1-5)
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
    prediction      SMALLINT    NOT NULL,   -- 0 or 1
    probability     REAL        NOT NULL,   -- 0.0 to 1.0

    -- Matched location
    matched_lat     DOUBLE PRECISION,
    matched_lon     DOUBLE PRECISION,
    place_name      TEXT,

    -- Agent outputs
    reasoning_source    TEXT,               -- 'huggingface' or 'template'
    sensitivity_results JSONB,              -- list of sensitivity check results
    flagged_conflicts   TEXT[]              -- array of conflict descriptions
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    n_queries   INTEGER     NOT NULL DEFAULT 1
);
"""

# Index on timestamp for fast drift queries like:
#   SELECT AVG(rainfall) FROM predictions
#   WHERE timestamp > NOW() - INTERVAL '7 days'
CREATE_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions (timestamp DESC);
"""

# Index on session_id for fast session lookups
CREATE_SESSION_INDEX = """
CREATE INDEX IF NOT EXISTS idx_predictions_session
    ON predictions (session_id);
"""


def init_db():
    """
    Create all tables and indexes.
    Safe to run multiple times — uses IF NOT EXISTS throughout.
    """
    print("Connecting to PostgreSQL...")
    conn = get_connection()
    cur  = conn.cursor()

    print("Creating predictions table...")
    cur.execute(CREATE_PREDICTIONS_TABLE)

    print("Creating sessions table...")
    cur.execute(CREATE_SESSIONS_TABLE)

    print("Creating indexes...")
    cur.execute(CREATE_TIMESTAMP_INDEX)
    cur.execute(CREATE_SESSION_INDEX)

    cur.close()
    conn.close()

    print("\nDatabase initialised successfully.")
    print("Tables: predictions, sessions")
    print("You can now start the app:  python app.py")


if __name__ == "__main__":
    init_db()