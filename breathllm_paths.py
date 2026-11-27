"""Local locations for downloaded assets and generated research records."""

from pathlib import Path
import os

PROJECT = Path(__file__).resolve().parent


def local_root(variable, default):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else PROJECT / default


DATA = local_root("BREATHLLM_DATA_ROOT", "data/ucddb")
WEIGHTS = local_root("BREATHLLM_WEIGHTS_ROOT", "weights")
OUTPUT = local_root("BREATHLLM_OUTPUT_ROOT", "outputs")
MANIFEST = OUTPUT / "audit/window_manifest.csv"
SPLIT = PROJECT / "breath5/experiments/paper_eval/nested_split.csv"
ASSETS = PROJECT / "config/外部资产.json"
