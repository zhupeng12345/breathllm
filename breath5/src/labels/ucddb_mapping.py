"""Breath5 class order and raw label mapping."""

CLASS_NAMES = ("Normal", "OSA", "CSA", "MSA", "Hypopnea")
IGNORE = -1
LABEL_VERSION = "breath5-sleep-midpoint-v1"
RAW_CLASSES = {
    "O": 1, "APNEA-O": 1, "C": 2, "APNEA-C": 2, "M": 3, "APNEA-M": 3,
    "HYP": 4, "HYP-O": 4, "HYP-C": 4, "HYP-M": 4,
    "PB EVENT": IGNORE, "PB": IGNORE, "CS": IGNORE,
    "CHEYNE-STOKES": IGNORE, "POSSIBLE": IGNORE,
}


def map_event(raw: str) -> int:
    return RAW_CLASSES[raw.strip().upper()]
