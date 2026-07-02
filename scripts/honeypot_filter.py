#!/usr/bin/env python3
"""
Honeypot filter — Step 1 of the ranking pipeline.

Scores each candidate on how "honeypot-like" they look. Higher score = more suspicious.
A honeypot is a profile that looks plausible at a glance but has internal
contradictions or platform-signal patterns that a real user would not produce.

The score is decomposed into named checks so downstream steps can:
  - subtract this score from the final ranking score,
  - or hard-reject candidates above a threshold before further scoring.

Only redrob_signals-based checks live here (per solution_scope.md step 1).
Content/keyword-stuffing checks belong in later filter stages.

Usage:
    python honeypot_filter.py --in <candidates.jsonl> --out <flagged.jsonl>
        [--sample N] [--threshold F] [--summary]

Output JSONL rows:
    {"candidate_id": ..., "honeypot_score": float, "reasons": [str, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterator


# "Today" — the challenge dataset is dated around mid-2026. We anchor to the
# current session date so future-date checks stay meaningful even if the file
# is re-run later. Override via --today if you want reproducibility.
DEFAULT_TODAY = date(2026, 7, 1)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def score_candidate(candidate: dict[str, Any], today: date) -> tuple[float, list[str]]:
    """Return (honeypot_score, reasons). Score in roughly 0..10+, unbounded above."""
    s = candidate.get("redrob_signals", {}) or {}
    p = candidate.get("profile", {}) or {}
    score = 0.0
    reasons: list[str] = []

    # ---- HARD IMPOSSIBILITIES (weight 3.0 each) --------------------------------
    # These are internal contradictions a genuine user cannot produce.

    signup = _parse_date(s.get("signup_date"))
    active = _parse_date(s.get("last_active_date"))
    if signup and active and active < signup:
        score += 3.0
        reasons.append("last_active before signup")
    if signup and signup > today:
        score += 3.0
        reasons.append("signup date in future")
    if active and active > today:
        score += 3.0
        reasons.append("last_active in future")

    sal = s.get("expected_salary_range_inr_lpa") or {}
    smin, smax = sal.get("min"), sal.get("max")
    if isinstance(smin, (int, float)) and isinstance(smax, (int, float)) and smin > smax:
        score += 3.0
        reasons.append(f"salary min>max ({smin}>{smax})")

    # Career_history duration_months must sum to roughly years_of_experience.
    # Empirically on the full dataset 99.95% of profiles match within ±0.5 yrs;
    # anything beyond ±1.5 yrs is a deliberate inconsistency. We don't credit
    # overlapping roles because the generator produces sequential careers only.
    yoe = p.get("years_of_experience")
    history = candidate.get("career_history") or []
    if isinstance(yoe, (int, float)) and history:
        sum_years = sum((r.get("duration_months") or 0) for r in history) / 12
        diff = sum_years - yoe
        if abs(diff) > 3.0:
            score += 3.0
            reasons.append(
                f"career sum {sum_years:.1f}y vs yoe {yoe} (Δ{diff:+.1f}y)"
            )
        elif abs(diff) > 1.5:
            score += 1.5
            reasons.append(
                f"career sum {sum_years:.1f}y vs yoe {yoe} (Δ{diff:+.1f}y)"
            )

    # ---- STRONG SIGNALS (weight 1.0-2.0) ---------------------------------------

    # Long dormant on platform. Anyone who hasn't logged in for a year is either
    # abandoned or a seeded profile.
    if active:
        days_since = (today - active).days
        if days_since > 365:
            score += 2.0
            reasons.append(f"inactive {days_since}d")
        elif days_since > 180:
            score += 0.5
            reasons.append(f"inactive {days_since}d")

    # No verification of any channel — real hiring candidates verify at least one.
    verified_channels = sum(
        bool(s.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected")
    )
    if verified_channels == 0:
        score += 2.0
        reasons.append("no email/phone/linkedin verified")
    elif verified_channels == 1:
        score += 0.3
        reasons.append("only 1 verification channel")

    # Zero real activity across every engagement axis. A live user leaves at least
    # one non-zero footprint over 30 days.
    activity_axes = [
        s.get("profile_views_received_30d", 0),
        s.get("applications_submitted_30d", 0),
        s.get("search_appearance_30d", 0),
        s.get("saved_by_recruiters_30d", 0),
        s.get("connection_count", 0),
    ]
    if all((v or 0) == 0 for v in activity_axes):
        score += 2.0
        reasons.append("zero activity on every axis")

    # Applied to zero roles AND recruiters never surfaced them AND they claim
    # open-to-work. A real open-to-work user drives at least one of these.
    if (
        s.get("open_to_work_flag")
        and (s.get("applications_submitted_30d") or 0) == 0
        and (s.get("search_appearance_30d") or 0) == 0
    ):
        score += 1.0
        reasons.append("open-to-work but zero apps and zero search appearances")

    # ---- SOFT SIGNALS (weight 0.3-0.8) -----------------------------------------

    # Bizarre response-time / response-rate combos.
    rrr = s.get("recruiter_response_rate")
    rt = s.get("avg_response_time_hours")
    if isinstance(rrr, (int, float)) and isinstance(rt, (int, float)):
        # Claims high response rate but never actually responds fast → suspicious.
        if rrr >= 0.7 and rt >= 240:
            score += 0.5
            reasons.append(f"high response_rate ({rrr}) but slow response ({rt}h)")
        # Claims almost never responds but response time is unreasonably fast →
        # sampling artifact of a synthetic profile.
        if rrr <= 0.1 and 0 < rt < 6:
            score += 0.5
            reasons.append(f"low response_rate ({rrr}) but very fast response ({rt}h)")

    # Interview_completion_rate = 0 despite non-trivial platform tenure.
    icr = s.get("interview_completion_rate")
    if signup and (today - signup).days > 365 and isinstance(icr, (int, float)) and icr == 0:
        score += 0.3
        reasons.append("tenured account with zero interview completions")

    # Endorsements without any connections — endorsements on Redrob require
    # connections, so this is a data-generation artifact.
    endorsements = s.get("endorsements_received") or 0
    conns = s.get("connection_count") or 0
    if endorsements > 20 and conns == 0:
        score += 1.0
        reasons.append(f"{endorsements} endorsements but 0 connections")

    # Profile completeness < 25 with high salary expectations — recruiters
    # rarely fund a bare profile at senior comp. Weak signal.
    pcs = s.get("profile_completeness_score")
    if isinstance(pcs, (int, float)) and pcs < 25 and isinstance(smax, (int, float)) and smax > 25:
        score += 0.3
        reasons.append(f"low completeness ({pcs}) but high salary ceiling ({smax} LPA)")

    return score, reasons


def iter_candidates(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream candidates from JSONL, or a JSON list if the file is a `.json`."""
    if path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        for i, c in enumerate(data):
            if limit is not None and i >= limit:
                return
            yield c
        return

    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="Input .jsonl or .json")
    ap.add_argument("--out", dest="out", required=True, help="Output .jsonl with scores")
    ap.add_argument("--sample", type=int, default=None, help="Only process first N rows")
    ap.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Score >= threshold is considered a honeypot (default 3.0)",
    )
    ap.add_argument(
        "--today",
        default=DEFAULT_TODAY.isoformat(),
        help=f"Anchor date for future/inactive checks (default {DEFAULT_TODAY})",
    )
    ap.add_argument("--summary", action="store_true", help="Print summary stats to stderr")
    args = ap.parse_args()

    today = date.fromisoformat(args.today)
    in_path = Path(args.inp)
    out_path = Path(args.out)

    total = 0
    flagged = 0
    reason_counts: dict[str, int] = {}
    score_buckets = {"0": 0, "0-1": 0, "1-3": 0, "3-6": 0, "6+": 0}

    with out_path.open("w") as out_f:
        for c in iter_candidates(in_path, args.sample):
            total += 1
            score, reasons = score_candidate(c, today)

            for r in reasons:
                key = r.split(" (")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1

            if score == 0:
                score_buckets["0"] += 1
            elif score < 1:
                score_buckets["0-1"] += 1
            elif score < 3:
                score_buckets["1-3"] += 1
            elif score < 6:
                score_buckets["3-6"] += 1
            else:
                score_buckets["6+"] += 1

            is_honeypot = score >= args.threshold
            if is_honeypot:
                flagged += 1

            out_f.write(
                json.dumps(
                    {
                        "candidate_id": c.get("candidate_id"),
                        "honeypot_score": round(score, 2),
                        "is_honeypot": is_honeypot,
                        "reasons": reasons,
                    }
                )
                + "\n"
            )

    if args.summary:
        pct = (flagged / total * 100) if total else 0
        print(f"Processed {total} candidates", file=sys.stderr)
        print(f"Flagged {flagged} ({pct:.1f}%) at threshold {args.threshold}", file=sys.stderr)
        print("Score buckets:", file=sys.stderr)
        for k, v in score_buckets.items():
            print(f"  {k}: {v}", file=sys.stderr)
        print("Top reasons:", file=sys.stderr)
        for r, n in sorted(reason_counts.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {n:6d}  {r}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
