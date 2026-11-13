"""Filesystem locations. Everything downloaded or generated lives under data/ (gitignored)."""

from __future__ import annotations

from lumina_train.model_contract import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
HDRI_DIR = DATA_DIR / "hdri"
POLYHAVEN_DIR = HDRI_DIR / "polyhaven"
AMBIENTCG_DIR = HDRI_DIR / "ambientcg"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = DATA_DIR / "runs"

SPLIT_PATH = REPO_ROOT / "training" / "splits" / "split.json"
EVAL_DIR = REPO_ROOT / "eval"
DOCS_DIR = REPO_ROOT / "docs"
MEDIA_DIR = DOCS_DIR / "media"
MODEL_META_PATH = REPO_ROOT / "web" / "public" / "models" / "lumina-sh.json"
SAMPLES_DIR = REPO_ROOT / "web" / "public" / "samples"
