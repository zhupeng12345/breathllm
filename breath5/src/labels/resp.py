"""解析 UCDDB 原始 RESP 事件记录及跨午夜时钟。"""

from datetime import time
import re

import numpy as np

KINDS = {
    "APNEA-C": "CSA", "APNEA-O": "OSA", "APNEA-M": "Mixed",
    "HYP-O": "Hypopnea", "HYP-C": "Hypopnea", "HYP-M": "Hypopnea",
    "PB EVENT": "Periodic", "POSSIBLE": "Uncertain",
}


CLOCK = re.compile(r"^\d{2}:\d{2}:\d{2}")


def clock_seconds(value):
    clock = time.fromisoformat(value)
    return clock.hour * 3600 + clock.minute * 60 + clock.second


def read_events(path, record, clock_zero, duration, stages):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="latin1").splitlines(), 1):
        if not CLOCK.match(line):
            if line_number > 3 and line.strip(" \x1a"):
                raise ValueError(f"{path}:{line_number}: unparsed annotation {line!r}")
            continue
        original = line[10:18].strip()
        period = line[18:25].strip()
        duration_sec = float(line[25:34])
        if not np.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError(f"{path}:{line_number}: invalid duration")
        start = (clock_seconds(line[:8]) - clock_zero) % 86400
        end = start + duration_sec
        valid = 0 <= start < end <= duration
        stage = int(stages[start // 30]) if start // 30 < len(stages) else -1
        low = line[34:42].strip()
        rows.append(dict(subject_id=record, record_id=record,
                         event_id=f"{record}-E{line_number:04d}",
                         source_file=path.name, source_line=line_number,
                         clock_time=line[:8], original_label=original,
                         category=KINDS.get(original, "Uncertain"), periodic_flag=period,
                         start_sec=start, end_sec=end, duration_sec=duration_sec,
                         start_stage=stage, in_record=valid,
                         annotated_spo2_low=float(low) if low else np.nan))
    return rows
