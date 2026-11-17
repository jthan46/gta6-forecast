"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Entry for make.fun's gta6-twitch-ccu-forecast-01 (program/v1, Python 3.13,
standard library only, no network, deterministic).

Idea: a launch day has a characteristic shape, and the case itself carries
the evidence — `comparables`, dozens of past big launches with their hourly
Twitch curves over their own release days. For each comparable this program
takes the day's hourly viewers relative to that day's first hour, in log
space:

    r_c(h) = ln(viewers_c(h) / viewers_c(1))          h = 2..24

and forecasts the case's hours as its known first-hour print p1 scaled by
quantiles of those log-ratios:

    q_tau(h) = p1 * exp(Q_tau({r_c(h)}))

The quantiles are weighted, because not every past launch is the same kind
of launch. A comparable counts more when

  * its first hour was about the target's size (a Gaussian kernel on
    log first-hour viewers): a launch whose first hour is already its peak
    decays into the day, one whose first hour is a pre-unlock trickle
    explodes when the unlock lands, and the two shapes must not be averaged
    together;
  * its unlock had the same shape as the target's (rolling midnight-local
    vs one simultaneous global moment vs a mix), read from the unlock notes
    each comparable carries and from the case's own notes for the target;
  * it is a clean launch: the comparables label their own tier, and
    early-access-contaminated or weak analogues count less.

The weighted quantiles are smoothed lightly across neighbouring hours. A
comparable is only used when its release day is actually in its series (a
first hour and most of the day present); the one comparable that is the
target category's own pre-release history has no launch day and drops out
by that rule. Every choice above was set by leaving each comparable out of
the pool and forecasting its day instead (eval_loo.py): the result beats
the house persistence baseline by over half on that test, and by about a
quarter on the big, clean, rolling-midnight launches that resemble this
one.

The grader runs `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), one case
in /input at a time; locally MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand
in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))

# Tunable shape of the forecast; eval_loo.py measures every choice here by
# leaving each comparable out of the pool and forecasting its day instead.
PARAMS = {
    "size_bw": 1.0,  # Gaussian kernel bandwidth on log first-hour viewers
    "class_same": 2.0,  # weight multiplier when unlock shapes match
    "class_unknown": 1.0,  # ... when either side's shape is unreadable
    "class_diff": 0.5,  # ... when they clearly differ
    "tier_weights": {"comparable": 1.0, "early-access": 0.5, "weak": 0.25},
    "smooth_passes": 2,  # [1, 2, 1] / 4 smoothing passes over hours, per level
    "spread_scale": 1.0,  # multiplies each log-quantile's distance from the median
    "min_day_hours": 12,  # a comparable needs at least this many release-day hours
    "min_first_hour": 1.0,  # and a first hour above this many viewers
}


def _epoch(ts: str) -> float:
    """Seconds since the epoch for an ISO 8601 UTC timestamp."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def release_day(comp: dict) -> dict[int, float]:
    """The comparable's release-day viewers by hour (1..24), from its hourly series.

    Points are whole-hour buckets starting at day_start; anything off the
    hour or outside the day is ignored, as is a non-positive viewer count.
    """
    start = _epoch(comp["day_start"])
    day: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        offset = (_epoch(point["from"]) - start) / 3600.0
        hour = int(offset)
        if offset == hour and 0 <= hour < 24:
            viewers = point.get("viewers")
            if viewers is not None and viewers > 0:
                day[hour + 1] = float(viewers)
    return day


def log_ratios(day: dict[int, float]) -> dict[int, float] | None:
    """log(viewers_h / viewers_1) for a release day, or None when unusable.

    None when the day has no first hour worth the name, or too few of its
    hours were sampled — the pre-release history comparable fails here,
    having no launch day at all.
    """
    first = day.get(1)
    if first is None or first < PARAMS["min_first_hour"]:
        return None
    if len(day) < PARAMS["min_day_hours"]:
        return None
    base = math.log(first)
    return {h: math.log(v) - base for h, v in day.items() if h >= 2}


def tier_of(unlock_text: str) -> str:
    """The tier a comparable labels itself with in its unlock notes."""
    text = unlock_text.lower()
    if "early-access-contaminated" in text:
        return "early-access"
    if "tier: weak" in text:
        return "weak"
    return "comparable"


def shape_of(text: str) -> str:
    """The unlock shape a text describes: rolling, simult(aneous), mixed, unknown."""
    t = text.lower()
    simult = any(
        marker in t
        for marker in ("simultaneous", "single global", "global unlock", "one global")
    )
    rolling = "midnight local" in t or "rolling" in t
    if simult and rolling:
        return "mixed"
    if simult:
        return "simult"
    if rolling:
        return "rolling"
    return "unknown"


def comparable_meta(comp: dict, day: dict[int, float]) -> dict:
    """What the weighting needs to know about one comparable."""
    unlock = comp.get("unlock") or ""
    return {
        "log_first": math.log(day[1]),
        "tier": tier_of(unlock),
        "shape": shape_of(unlock),
    }


def pool_weight(profile: dict, meta: dict) -> float:
    """How much one comparable counts for a target with the given profile."""
    weight = math.exp(
        -((meta["log_first"] - profile["log_first"]) ** 2)
        / (2.0 * PARAMS["size_bw"] ** 2)
    )
    weight *= PARAMS["tier_weights"].get(meta["tier"], 1.0)
    if meta["shape"] == profile["shape"]:
        weight *= PARAMS["class_same"]
    elif meta["shape"] == "unknown" or profile["shape"] == "unknown":
        weight *= PARAMS["class_unknown"]
    else:
        weight *= PARAMS["class_diff"]
    return weight


def _weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """The tau quantile of (value, weight) pairs.

    Each value sits at the midpoint of its own slice of the total weight
    and the quantile interpolates linearly between neighbours, so the
    extreme levels stay inside the observed range.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if len(pairs) == 1 or total <= 0.0:
        return pairs[0][0]
    marks = []
    cum = 0.0
    for value, w in pairs:
        cum += w
        marks.append((cum - w / 2.0) / total)
    if tau <= marks[0]:
        return pairs[0][0]
    if tau >= marks[-1]:
        return pairs[-1][0]
    for i in range(1, len(marks)):
        if tau <= marks[i]:
            frac = (tau - marks[i - 1]) / (marks[i] - marks[i - 1])
            return pairs[i - 1][0] + frac * (pairs[i][0] - pairs[i - 1][0])
    return pairs[-1][0]


def _smooth(curves: list[list[float]]) -> list[list[float]]:
    """Light [1, 2, 1] / 4 smoothing of each level's curve across hours."""
    for _ in range(PARAMS["smooth_passes"]):
        for level in range(len(LEVELS)):
            row = [curves[i][level] for i in range(len(HOURS))]
            out = row[:]
            for i in range(1, len(row) - 1):
                out[i] = (row[i - 1] + 2.0 * row[i] + row[i + 1]) / 4.0
            for i in range(len(HOURS)):
                curves[i][level] = out[i]
    return curves


def forecast_from_entries(
    p1: float, profile: dict, entries: list[tuple[dict[int, float], dict]]
) -> dict[int, list[float]]:
    """The forecast rows {hour: [q05, q25, q50, q75, q95]} for a first-hour print p1.

    `entries` is the pool as (log-ratios, meta) pairs; `profile` describes
    the target the same way meta describes a comparable.
    """
    weighted = [(ratios, pool_weight(profile, meta)) for ratios, meta in entries]
    curves: list[list[float]] = []
    last: list[float] | None = None
    for hour in HOURS:
        pairs = [(ratios[hour], w) for ratios, w in weighted if hour in ratios]
        if pairs:
            last = [_weighted_quantile(pairs, tau) for tau in LEVELS]
        curves.append(list(last) if last is not None else [0.0] * len(LEVELS))
    curves = _smooth(curves)
    scale = PARAMS["spread_scale"]
    result: dict[int, list[float]] = {}
    for i, hour in enumerate(HOURS):
        median = curves[i][2]
        qs = []
        for lq in curves[i]:
            value = p1 * math.exp(median + scale * (lq - median))
            value = min(max(value, 0.01), 1_000_000_000.0)
            qs.append(round(value, 2))
        for j in range(1, len(qs)):  # the row must be non-decreasing
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        result[hour] = qs
    return result


def pool_entries(case: dict) -> list[tuple[dict[int, float], dict]]:
    """The case's usable comparables as (log-ratios, meta) pairs."""
    entries = []
    for comp in case.get("comparables") or []:
        day = release_day(comp)
        ratios = log_ratios(day)
        if ratios is not None:
            entries.append((ratios, comparable_meta(comp, day)))
    return entries


def target_profile(case: dict, p1: float) -> dict:
    """The target described the way comparable_meta describes a comparable.

    The unlock shape is read from the case's own notes, which say when the
    game really unlocks; an unreadable shape simply turns the shape
    weighting off (every comparable gets the unknown multiplier).
    """
    return {"log_first": math.log(p1), "shape": shape_of(case.get("notes") or "")}


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    rows = forecast_from_entries(p1, target_profile(case, p1), pool_entries(case))
    return {"hours": [{"hour": row["hour"], "q": rows[row["hour"]]} for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
