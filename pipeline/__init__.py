"""HiveSpace desk occupancy pipeline package.

Exposes `run_pipeline` so the whole pipeline can be executed via:
    python pipeline/init.py
or:
    python pipeline/__init__.py
or imported with:
    from pipeline import run_pipeline
"""

import sys
from pathlib import Path

# Ensure the pipeline directory is on sys.path
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    from .init import run_pipeline
except ImportError:
    from init import run_pipeline

__all__ = ["run_pipeline"]

if __name__ == "__main__":
    run_pipeline()
