"""Leave-one-out evaluation of main.py's forecaster over a case's comparables.

Each comparable with a complete release day takes a turn as the target: its
own first hour is the known print, the pool is every other usable comparable
in the same case, and main.py's exact forecasting code (imported, not copied)
predicts hours 2..24. The score is the contest's own: mean pinball loss on
the natural-log scale over the target's printed hours x the five levels. The
house persistence baseline is scored the same way beside it.

    python3 eval_loo.py <case.in> [<case.in> ...]

Prints one line per case and a pooled line: the model's mean loss, the
baseline's, and the margin between them.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("entry", HERE / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

LEVELS = entry.LEVELS
Z = [-1.645, -0.674, 0, 0.674, 1.645]  # the house baseline's normal quantiles
BASE_SIGMA = 0.15  # the house baseline's spread per square-root hour


def pinball(y: float, q: float, tau: float) -> float:
    d = math.log(max(y, 1.0)) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def baseline_row(p1: float, hour: int) -> list[float]:
    return [p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1)) for z in Z]


def evaluate(case: dict) -> tuple[float, float, int, int]:
    """(model loss, baseline loss, targets, items) for one case's comparables."""
    pool_all = []
    for comp in case["comparables"]:
        day = entry.release_day(comp)
        ratios = entry.log_ratios(day)
        if ratios is not None:
            pool_all.append((comp, day, ratios, entry.comparable_meta(comp, day)))
    model_total = base_total = 0.0
    items = 0
    for comp, day, _ratios, _meta in pool_all:
        p1 = day[1]
        # The target described exactly as forecast_case describes a case:
        # its first hour, and its unlock shape from its own notes.
        profile = {"log_first": math.log(p1), "shape": entry.shape_of(comp.get("unlock") or "")}
        entries = [(r, m) for c2, _d2, r, m in pool_all if c2 is not comp]
        forecast = entry.forecast_from_entries(p1, profile, entries)
        for hour in range(2, 25):
            if hour not in day:
                continue
            y = day[hour]
            for tau, q_model, q_base in zip(LEVELS, forecast[hour], baseline_row(p1, hour)):
                model_total += pinball(y, q_model, tau)
                base_total += pinball(y, q_base, tau)
                items += 1
    return model_total / items, base_total / items, len(pool_all), items


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: eval_loo.py <case.in> [<case.in> ...]")
    pooled_model = pooled_base = 0.0
    pooled_items = 0
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        model, base, targets, items = evaluate(case)
        print(
            f"{case['id']}: model {model:.6f}  baseline {base:.6f}  "
            f"margin {100.0 * (base - model) / base:+.1f}%  ({targets} targets, {items} items)"
        )
        # Recompute pooled sums from the means (items are the weights).
        pooled_model += model * items
        pooled_base += base * items
        pooled_items += items
    if len(paths) > 1:
        model = pooled_model / pooled_items
        base = pooled_base / pooled_items
        print(
            f"pooled: model {model:.6f}  baseline {base:.6f}  "
            f"margin {100.0 * (base - model) / base:+.1f}%  ({pooled_items} items)"
        )


if __name__ == "__main__":
    main()
