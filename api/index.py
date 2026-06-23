"""
api/index.py
Entry point for Vercel's Python (WSGI) serverless runtime.
Vercel auto-detects an `app` WSGI callable in this file.

NOTE: Vercel's serverless functions have a deployment size limit and a
cold-start cost for heavier scientific Python packages (pandas, numpy,
scikit-learn). This works for light/demo traffic, but for the full model
+ ~3,600-point reference dataset used here, Render (Docker or native
Python, see render.yaml/Dockerfile) is the better fit and is the
recommended deployment target. This file is provided so Vercel remains
an option if you trim the reference dataset or move it to an external
store.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app  # noqa: E402

# Vercel's Python runtime looks for a module-level WSGI app named `app`
