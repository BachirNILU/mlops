"""The agent's tools. Plain Python functions with a Pydantic input schema and a permission level.

Design rule: the LLM never computes numbers. It asks a tool, and the tool computes them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from lab.config import DATA, RUNS
from lab.schemas import AnalyzeSensorInput, CreateWorkOrderInput, SearchDocsInput, WorkOrdersInput

LIMITS = {  # from the manuals (MAN-T-01, MAN-T-02, MAN-P-05); a real system would read them from a database
    ("T1", "bearing_temp_c"): {"normal_max": 60, "alarm": 70, "unit": "degC"},
    ("T2", "bearing_temp_c"): {"normal_max": 60, "alarm": 70, "unit": "degC"},
    ("P-101", "bearing_temp_c"): {"normal_max": 55, "alarm": 75, "unit": "degC"},
    ("T1", "vibration_mm_s"): {"normal_max": 4.5, "alarm": 7.1, "unit": "mm/s"},
    ("T2", "vibration_mm_s"): {"normal_max": 4.5, "alarm": 7.1, "unit": "mm/s"},
    ("P-101", "vibration_mm_s"): {"normal_max": 4.5, "alarm": 7.1, "unit": "mm/s"},
}


# ---------------------------------------------------------------- tool functions


def search_docs(query: str, k: int = 4) -> list[dict]:
    """Hybrid search + rerank over the maintenance documents (see lab/rag.py)."""
    from lab.rag import get_retriever

    hits = get_retriever().search(query, k=k)
    return [
        {
            "source_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "section": c.section,
            "text": c.text,
            "score": round(s, 3),
        }
        for c, s in hits
    ]


def _load_sensors() -> pd.DataFrame:
    path = DATA / "sensors.csv"
    if not path.exists():
        raise FileNotFoundError("data/sensors.csv missing. Run: python scripts/00_make_data.py")
    return pd.read_csv(path, parse_dates=["timestamp"])


def analyze_sensor(asset_id: str, signal: str, days: int = 14) -> dict:
    """Trend + anomaly check on one signal. Your time-series skills live here.

    - baseline: the 30 days before the recent window
    - z-score of the latest daily mean against the baseline
    - linear trend over the window, with a 95% confidence interval on the slope
    - change start: first day where the daily mean leaves baseline mean + 3 std
    """
    df = _load_sensors()
    s = df[df.asset_id == asset_id].set_index("timestamp")[signal].dropna()
    if s.empty:
        raise ValueError(f"No data for {asset_id}/{signal}")
    daily = s.resample("D").mean()
    recent, baseline = daily.iloc[-days:], daily.iloc[-days - 30 : -days]
    mu, sd = baseline.mean(), baseline.std(ddof=1)
    z = (recent.iloc[-1] - mu) / sd

    x = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent.values, 1)
    resid = recent.values - (slope * x + intercept)
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    ci = (slope - 1.96 * se, slope + 1.96 * se)  # normal approx.; fine for a toy

    out_of_band = recent[recent > mu + 3 * sd]
    lim = LIMITS[(asset_id, signal)]
    last = float(s.iloc[-1])
    status = (
        "alarm"
        if last >= lim["alarm"]
        else "warning"
        if (last > lim["normal_max"] or abs(z) > 3)
        else "normal"
    )
    return {
        "source_id": f"tool:analyze_sensor:{asset_id}:{signal}",
        "asset_id": asset_id,
        "signal": signal,
        "unit": lim["unit"],
        "last_value": round(last, 2),
        "last_daily_mean": round(float(recent.iloc[-1]), 2),
        "baseline_mean": round(float(mu), 2),
        "baseline_std": round(float(sd), 2),
        "z_score": round(float(z), 1),
        "trend_per_day": round(float(slope), 2),
        "trend_ci95": [round(float(ci[0]), 2), round(float(ci[1]), 2)],
        "change_start": str(out_of_band.index[0].date()) if len(out_of_band) else None,
        "normal_max": lim["normal_max"],
        "alarm_limit": lim["alarm"],
        "status": status,
        "data_end": str(s.index[-1]),
    }


def get_work_orders(asset_id: str) -> dict:
    """Long-term memory: past repairs for one asset."""
    wos = json.loads((DATA / "work_orders.json").read_text())
    items = [w for w in wos if w["asset_id"] == asset_id]
    return {
        "source_id": f"tool:get_work_orders:{asset_id}",
        "asset_id": asset_id,
        "work_orders": items,
    }


def create_work_order(asset_id: str, description: str, priority: str) -> dict:
    """WRITE action. Only runs after a human approves it (see lab/guards.py)."""
    RUNS.mkdir(exist_ok=True)
    path = RUNS / "proposed_work_orders.json"
    items = json.loads(path.read_text()) if path.exists() else []
    wo = {
        "id": f"WO-DRAFT-{len(items) + 1:03d}",
        "asset_id": asset_id,
        "description": description,
        "priority": priority,
        "status": "proposed (not released)",
    }
    items.append(wo)
    path.write_text(json.dumps(items, indent=2))
    return {"source_id": "tool:create_work_order", **wo}


# ---------------------------------------------------------------- registry


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[..., Any]
    permission: str = "read"  # "read" runs freely; "write" needs human approval

    def json_schema(self) -> dict:
        return self.input_model.model_json_schema()

    def run(self, args: dict) -> Any:
        validated = self.input_model.model_validate(args)  # never trust LLM arguments blindly
        return self.fn(**validated.model_dump())


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_docs",
            "Search maintenance manuals, procedures, inspection reports and logs. "
            "Returns passages with a source_id to cite.",
            SearchDocsInput,
            search_docs,
        ),
        Tool(
            "analyze_sensor",
            "Compute trend, z-score and status for one sensor signal of one asset. "
            "Use it for any question about temperatures, vibration or trends.",
            AnalyzeSensorInput,
            analyze_sensor,
        ),
        Tool(
            "get_work_orders",
            "List past repairs and inspections for one asset.",
            WorkOrdersInput,
            get_work_orders,
        ),
        Tool(
            "create_work_order",
            "Propose a new work order. A human must approve it.",
            CreateWorkOrderInput,
            create_work_order,
            permission="write",
        ),
    ]
}

READ_ONLY = [t for t in TOOLS.values() if t.permission == "read"]
