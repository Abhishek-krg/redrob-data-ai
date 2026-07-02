#!/usr/bin/env python3
"""
Weighted blender that consumes the first-level output of ranking.py and
appends per-signal + final_score columns to each candidate.

Contract:
  input  = list of candidate dicts already scored by ranking.py. Each dict is
           expected to carry:
              - candidate_id, project_match_score, per_project_evidence
              - the ranking-signal columns from the LanceDB `candidates` table
                (recruiter_response_rate, saved_by_recruiters_30d,
                 github_activity_score, interview_completion_rate,
                 profile_completeness_score, open_to_work, endorsements_received,
                 connection_count, verified_email/phone, linkedin_connected,
                 skill_assessment_scores_json)
           No round-trip to candidates.jsonl is needed — everything lives on
           the row already.
  output = same list with additional columns:
              - <signal>_score for each signal we blend in (already normalized)
              - final_score = weighted sum of {project_match_score, all signals}
           Sorted by final_score descending, truncated to top_k.

Signals in the default weight sheet (see DEFAULT_WEIGHTS below):
  project_match_score       vector match from ranking.py (already in [0,1])
  recruiter_response_rate   0..1
  skill_assessment_score    mean of JD-required skill assessments; 0..1
  saved_by_recruiters_30d   normalized 0..1 in the pool
  github_activity_score     -1 treated as null; 0..1 in the pool
  interview_completion_rate 0..1
  profile_completeness      0..1 (raw 0-100 rescaled)
  open_to_work_flag         0 or 1
  endorsement_rate          endorsements / (connections + 1), normalized 0..1
  verification_score        0..1: sum(verified_email + verified_phone
                                       + linkedin_connected) / 3

Public API:
  blend(candidates, weights=None, jd_required_skills=None, top_k=100) -> list

CLI:
  python scripts/weighted_ranker.py \\
      --in top10_candidates_full.json \\
      --out top100_ranked.json --top-k 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Default weight sheet — comes from the earlier signal-priority discussion.
# The keys are the *_score column names the blender computes; values are
# relative importances (they don't need to sum to 1 — we normalize signals
# individually before combining).
# ---------------------------------------------------------------------------
# Fallback weight sheet — used only if scripts/weights.json is missing or
# unreadable. Keep this in sync with the file so behavior is well-defined
# even without the JSON. Editing the JSON is the intended path; the auto
# rank adjuster in scripts/auto_rank_adjuster.py rewrites it.
_FALLBACK_WEIGHTS: dict[str, float] = {
    "project_match_score": 0.42,          # semantic career-vs-projects match; dominant but not overwhelming
    "recruiter_response_rate_score": 0.15,# JD explicitly calls this out
    "skill_assessment_score": 0.08,       # objective + JD-aligned but sparse (few candidates take assessments for JD skills)
    "saved_by_recruiters_score": 0.08,    # market validation — other recruiters bookmarked them
    "tenure_at_product_score": 0.10,      # rewards long product-co tenure; caps at 8 yrs so early-career doesn't stall
    "github_activity_score_norm": 0.03,   # public-code signal for AI/ML roles
    "interview_completion_score": 0.10,   # conversion / do-they-show-up proxy
    "profile_completeness_score_norm": 0.00,
    "open_to_work_score": 0.02,           # small nudge; passive candidates aren't penalized much
    "endorsement_rate_score": 0.02,       # tie-breaker
    "verification_score": 0.00,           # gated by honeypot filter separately
    # NEGATIVE — down-weights candidates whose career is spent at excluded cos.
    # Score is fraction of career months there, so a full-career-at-TCS hire
    # gets -0.30; brief stints barely register.
    "excluded_company_penalty": -0.30,
}


WEIGHTS_PATH = Path(__file__).with_name("weights.json")


def load_weights(path: Path | None = None) -> dict[str, float]:
    """Load weights from JSON. Falls back to the hardcoded sheet on missing
    or malformed file (with a stderr warning) — never crashes at import."""
    p = path or WEIGHTS_PATH
    try:
        with p.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("weights.json must be a JSON object")
        return {k: float(v) for k, v in data.items()}
    except FileNotFoundError:
        print(f"[weighted_ranker] {p} not found — using fallback weights.", file=sys.stderr)
        return dict(_FALLBACK_WEIGHTS)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"[weighted_ranker] {p} unreadable ({e}) — using fallback weights.", file=sys.stderr)
        return dict(_FALLBACK_WEIGHTS)


# Populated at import time so callers can still do `from weighted_ranker import DEFAULT_WEIGHTS`.
DEFAULT_WEIGHTS: dict[str, float] = load_weights()


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _get_signals(candidate: dict) -> dict:
    """Extractors read signals off the LanceDB candidate row directly.
    If a legacy shape shows up (with a nested `full_profile.redrob_signals`),
    we fall back to that so old dumps still work with `weighted_ranker --in`.
    """
    if "recruiter_response_rate" in candidate or "profile_completeness_score" in candidate:
        return candidate
    fp = candidate.get("full_profile") or {}
    return fp.get("redrob_signals") or {}


def _minmax(values: list[float | None]) -> list[float]:
    """Min-max normalize into [0,1]. `None` entries pass through as 0.
    A degenerate range (all values equal) collapses to 0 so the signal
    contributes nothing (rather than everything, which would be misleading)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return [0.0] * len(values)
    lo, hi = min(valid), max(valid)
    span = hi - lo
    if span == 0:
        return [0.0] * len(values)
    return [(v - lo) / span if v is not None else 0.0 for v in values]


def _normalize_skill(name: str) -> str:
    return " ".join((name or "").lower().split())


# ---------------------------------------------------------------------------
# Per-signal feature extractors. Return one raw value per candidate, or None
# when the signal is not meaningful for that candidate (sentinel value etc.).
# ---------------------------------------------------------------------------
def _extract_recruiter_response(candidates: list[dict]) -> list[float | None]:
    return [
        float(_get_signals(c).get("recruiter_response_rate", 0.0) or 0.0)
        for c in candidates
    ]


def _extract_saved_by_recruiters(candidates: list[dict]) -> list[float | None]:
    return [
        float(_get_signals(c).get("saved_by_recruiters_30d", 0) or 0)
        for c in candidates
    ]


def _extract_github(candidates: list[dict]) -> list[float | None]:
    # -1 means no GitHub linked — treat as "no signal" (None) so it doesn't
    # push a scientist without public code below one with a low real score.
    out: list[float | None] = []
    for c in candidates:
        v = _get_signals(c).get("github_activity_score")
        if v is None or v < 0:
            out.append(None)
        else:
            out.append(float(v))
    return out


def _extract_interview_completion(candidates: list[dict]) -> list[float | None]:
    return [
        float(_get_signals(c).get("interview_completion_rate", 0.0) or 0.0)
        for c in candidates
    ]


def _extract_profile_completeness(candidates: list[dict]) -> list[float | None]:
    return [
        float(_get_signals(c).get("profile_completeness_score", 0.0) or 0.0)
        for c in candidates
    ]


def _extract_open_to_work(candidates: list[dict]) -> list[float | None]:
    # LanceDB row uses `open_to_work`; raw candidates.jsonl uses `open_to_work_flag`.
    out: list[float | None] = []
    for c in candidates:
        s = _get_signals(c)
        v = s.get("open_to_work")
        if v is None:
            v = s.get("open_to_work_flag")
        out.append(1.0 if v else 0.0)
    return out


def _extract_endorsement_rate(candidates: list[dict]) -> list[float | None]:
    # endorsements per connection is a much cleaner signal than raw endorsements,
    # which just rewards long-standing accounts. +1 in the denominator avoids
    # divide-by-zero and keeps candidates with 0 connections at 0.
    out: list[float | None] = []
    for c in candidates:
        s = _get_signals(c)
        endorsements = float(s.get("endorsements_received", 0) or 0)
        connections = float(s.get("connection_count", 0) or 0)
        out.append(endorsements / (connections + 1.0))
    return out


def _extract_product_tenure(
    candidates: list[dict], exclude_companies: list[str] | None
) -> list[float | None]:
    """Total months at NON-excluded companies, then clamped at 96 months (8 yrs)
    and divided by 96 to produce a [0,1] score.

    Rewards candidates with long product-company tenure. Excluded-company months
    are already captured by excluded_company_penalty; we don't double-count them
    here."""
    excluded = {c.strip().lower() for c in (exclude_companies or []) if c}
    out: list[float | None] = []
    CAP_MONTHS = 96  # 8 years — matches JD's sweet spot
    for c in candidates:
        career_raw = c.get("career_json") or "[]"
        try:
            career = json.loads(career_raw) if isinstance(career_raw, str) else career_raw
        except json.JSONDecodeError:
            career = []
        product_months = 0
        for r in career:
            company = (r.get("company") or "").strip().lower()
            if not company:
                continue
            if company in excluded:
                continue
            product_months += int(r.get("months") or 0)
        out.append(min(product_months, CAP_MONTHS) / CAP_MONTHS)
    return out


def _extract_excluded_company_penalty(
    candidates: list[dict], exclude_companies: list[str] | None
) -> list[float | None]:
    """Fraction of the candidate's total career months spent at any company
    in `exclude_companies`. Range [0, 1]:
        0.0 = never worked at an excluded company
        1.0 = entire career at excluded companies

    Combined with a NEGATIVE weight, this pulls down candidates whose career
    is dominated by JD-flagged companies (e.g. TCS/Infosys/… for the Redrob
    JD) proportionally to how much of their career is there. Brief stints
    barely register; single-company careers get the full penalty.

    Empty exclude_companies list -> 0.0 for everyone (no-op signal).
    """
    if not exclude_companies:
        return [0.0] * len(candidates)

    excluded = {c.strip().lower() for c in exclude_companies if c}
    out: list[float | None] = []
    for c in candidates:
        career_raw = c.get("career_json") or "[]"
        try:
            career = json.loads(career_raw) if isinstance(career_raw, str) else career_raw
        except json.JSONDecodeError:
            career = []
        total_months = 0
        excluded_months = 0
        for r in career:
            months = int(r.get("months") or 0)
            total_months += months
            company = (r.get("company") or "").strip().lower()
            if company and company in excluded:
                excluded_months += months
        out.append(excluded_months / total_months if total_months else 0.0)
    return out


def _extract_verification(candidates: list[dict]) -> list[float | None]:
    # 0..1: fraction of the three verification channels the candidate cleared.
    out: list[float | None] = []
    for c in candidates:
        s = _get_signals(c)
        cleared = sum(
            bool(s.get(k))
            for k in ("verified_email", "verified_phone", "linkedin_connected")
        )
        out.append(cleared / 3.0)
    return out


def _extract_skill_assessment(
    candidates: list[dict], jd_required_skills: list[str] | None
) -> list[float | None]:
    """Mean skill_assessment_scores for the JD's required skills, in [0,1].

    - If jd_required_skills is empty, this signal is 0 for everyone
      (i.e. it contributes nothing to the blend — safe no-op).
    - Missing an assessment for a required skill counts as 0 for that skill
      (a signal: they can't demonstrate the required skill on-platform).
    """
    out: list[float | None] = []
    if not jd_required_skills:
        return [0.0] * len(candidates)

    req_norm = [_normalize_skill(s) for s in jd_required_skills if s]
    if not req_norm:
        return [0.0] * len(candidates)

    for c in candidates:
        s = _get_signals(c)
        # LanceDB row: skill_assessment_scores_json (str). Raw jsonl: skill_assessment_scores (dict).
        assess = s.get("skill_assessment_scores")
        if assess is None:
            raw = s.get("skill_assessment_scores_json") or "{}"
            try:
                assess = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                assess = {}
        assess_norm = {_normalize_skill(k): float(v) for k, v in (assess or {}).items()}
        matched = [assess_norm.get(rs, 0.0) for rs in req_norm]
        mean = sum(matched) / len(matched) / 100.0  # bring 0-100 into 0-1
        out.append(mean)
    return out


# Map each blended score column to (weight_sheet_key, extractor_fn).
# The signals that are already in [0,1] semantically (rates, project match) do
# NOT get re-min-max-normalized — that would flatten meaningful differences
# down to the pool's spread. Only unbounded / integer signals get min-max'd.
_ALREADY_UNIT = {
    "project_match_score",
    "recruiter_response_rate_score",
    "skill_assessment_score",
    "interview_completion_score",
    "open_to_work_score",
    "verification_score",
    "profile_completeness_score_norm",  # /100 = same effect as scaling by pool span, but stable across pools
    "excluded_company_penalty",         # already a fraction in [0,1]
    "tenure_at_product_score",          # already normalized to [0,1] by the CAP inside the extractor
}


# ---------------------------------------------------------------------------
# The public function.
# ---------------------------------------------------------------------------
def blend(
    candidates: list[dict],
    weights: dict[str, float] | None = None,
    jd_required_skills: list[str] | None = None,
    jd_exclude_companies: list[str] | None = None,
    top_k: int = 100,
) -> list[dict]:
    """Attach per-signal scores + final_score to each candidate and return
    the top-K sorted by final_score descending.

    Args:
        candidates: output of ranking.py (list of dicts with full_profile)
        weights:    dict[column_name -> weight]. Missing keys default to 0
                    (so any DEFAULT_WEIGHTS not overridden still apply).
                    Extra keys are ignored.
        jd_required_skills: list of required skill names from the compiled JD.
                    When omitted, the skill_assessment_score column is 0 for
                    everyone (safe zero-contribution).
        jd_exclude_companies: list of companies to down-weight (e.g. from
                    jd.exclude_companies). Candidates get a proportional
                    penalty based on months at any listed company. When
                    omitted, this signal is 0 for everyone.
        top_k:      truncate the returned list to this many rows.
    """
    if not candidates:
        return []

    # Start from defaults, layer caller weights on top so callers can override
    # a subset without having to reproduce the whole sheet.
    effective_weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    # ---- Raw feature extraction ----
    raw: dict[str, list[float | None]] = {
        "project_match_score": [
            float(c.get("project_match_score") or 0.0) for c in candidates
        ],
        "recruiter_response_rate_score": _extract_recruiter_response(candidates),
        "skill_assessment_score": _extract_skill_assessment(candidates, jd_required_skills),
        "saved_by_recruiters_score": _extract_saved_by_recruiters(candidates),
        "github_activity_score_norm": _extract_github(candidates),
        "interview_completion_score": _extract_interview_completion(candidates),
        "profile_completeness_score_norm": [
            v / 100.0 for v in _extract_profile_completeness(candidates)
        ],
        "open_to_work_score": _extract_open_to_work(candidates),
        "endorsement_rate_score": _extract_endorsement_rate(candidates),
        "verification_score": _extract_verification(candidates),
        "excluded_company_penalty": _extract_excluded_company_penalty(
            candidates, jd_exclude_companies
        ),
        "tenure_at_product_score": _extract_product_tenure(
            candidates, jd_exclude_companies
        ),
    }

    # ---- Normalize unbounded signals to [0,1] within the pool ----
    normalized: dict[str, list[float]] = {}
    for col, values in raw.items():
        if col in _ALREADY_UNIT:
            normalized[col] = [float(v) if v is not None else 0.0 for v in values]
        else:
            normalized[col] = _minmax(values)

    # ---- Attach per-signal columns + compute final_score ----
    for i, cand in enumerate(candidates):
        signal_breakdown: dict[str, float] = {}
        total = 0.0
        for col, series in normalized.items():
            score = round(series[i], 6)
            cand[col] = score
            w = float(effective_weights.get(col, 0.0))
            signal_breakdown[col] = {"score": score, "weight": w, "contribution": round(score * w, 6)}
            total += score * w
        cand["score_breakdown"] = signal_breakdown
        cand["final_score"] = round(total, 6)

    # ---- Sort + truncate ----
    ranked = sorted(candidates, key=lambda c: c.get("final_score", 0.0), reverse=True)
    ranked = ranked[:top_k]

    # Re-number rank in the new order.
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
    return ranked


# ---------------------------------------------------------------------------
# CLI convenience for standalone runs (e.g. reblending an existing dump with
# different weights without rerunning the whole pipeline).
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="JSON produced by ranking.py")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--jd", default="scripts/jd_compiled.json",
                    help="Compiled JD JSON — used to pull required_technical_skills for the skill-assessment signal")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument(
        "--weights",
        default=None,
        help="Optional JSON string overriding weights (e.g. '{\"project_match_score\": 0.5}')",
    )
    args = ap.parse_args()

    with Path(args.inp).open() as f:
        candidates = json.load(f)

    weights = json.loads(args.weights) if args.weights else None

    jd_required_skills: list[str] = []
    jd_exclude_companies: list[str] = []
    try:
        with Path(args.jd).open() as f:
            jd = json.load(f)
        jd_required_skills = list(jd.get("required_technical_skills") or [])
        jd_exclude_companies = list(jd.get("exclude_companies") or [])
    except FileNotFoundError:
        print(f"[warn] no JD at {args.jd}; skill_assessment_score will be 0", file=sys.stderr)

    ranked = blend(
        candidates,
        weights=weights,
        jd_required_skills=jd_required_skills,
        jd_exclude_companies=jd_exclude_companies,
        top_k=args.top_k,
    )

    with Path(args.out).open("w") as f:
        json.dump(ranked, f, indent=2)
    print(f"Wrote {args.out} ({len(ranked)} rows)")
    for r in ranked[:10]:
        print(
            f"  #{r['rank']}  {r['candidate_id']}  "
            f"final={r['final_score']}  proj={r.get('project_match_score')}  "
            f"resp={r.get('recruiter_response_rate_score')}  "
            f"skill={r.get('skill_assessment_score')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())