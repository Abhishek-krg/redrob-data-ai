#!/usr/bin/env python3
"""
LLM-as-judge ranking auditor + weight tuner.

Runs the full ranking pipeline, hands the JD + current top-N profiles +
current weights to Qwen 3.7-plus (via Fireworks), and asks the LLM to:
  1. Name the candidates who are misplaced and explain why.
  2. Propose an updated weight sheet.

We apply the proposed weights (if `--apply`) and iterate until either:
  - --max-iters is hit, OR
  - the LLM's proposed weights have stabilized: the max per-signal delta
    across the last N iterations (default 3) is below --stability-threshold
    (default 0.02). That means the loop is converging on numbers rather than
    oscillating.

This deliberately doesn't rely on a hand-coded JD-fit checklist — the LLM
sees the raw JD text and full candidate profiles and forms its own
judgment. Every iteration is one LLM call (structured YAML output).

Usage:
    # Dry-run one iteration, print the LLM's judgment + proposed weights:
    python scripts/auto_rank_adjuster.py

    # Apply proposed weights to scripts/weights.json and iterate up to 5x:
    python scripts/auto_rank_adjuster.py --apply --max-iters 5

Environment: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME loaded from
scripts/.env by llm_JD_compiler._load_dotenv (already done at import).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import lancedb

# Loading .env before anything else so LLM_* env vars are populated.
from llm_JD_compiler import _load_dotenv, _parse_yaml_lenient  # type: ignore

_load_dotenv(Path(__file__).with_name(".env"))

import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402

from ranking import run as run_ranking  # noqa: E402
from utils import load_title_groups  # noqa: E402
from weighted_ranker import (  # noqa: E402
    DEFAULT_WEIGHTS,
    WEIGHTS_PATH,
    blend as weighted_blend,
    load_weights,
)


DEFAULT_FAMILIES = ["ai_ml_core", "data_science", "backend_engineering", "data_analytics"]


# ---------------------------------------------------------------------------
# Prompting.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior technical recruiter reviewing a candidate ranking system.

You will be given:
  - the full text of a Job Description (JD)
  - the current TOP-N candidates a ranking system has selected, WITH their profile details, career history, key signals, and the system's ranking score
  - the WEIGHT SHEET the ranking system is currently using — a dict of {signal_name: float}

Your job:
  1. Read the JD carefully. Note explicit dealbreakers, ideal-candidate description, must-haves, and cultural cues.
  2. Evaluate whether the current top-N is a good ranking for THIS JD. Focus on:
     - Are truly ideal candidates (JD's stated "6-8 yrs, product-co, shipped ranking/search systems" template) ranked at the top?
     - Are there candidates in the top-N who would obviously be rejected in a real interview (e.g. wrong domain, disqualified per JD's "we do NOT want" list, red flags in career history)?
     - Are there candidates in the middle/back of the top-N who are OBJECTIVELY BETTER than those at the top?
  3. If the ranking has issues, identify specific candidates by candidate_id and explain in one sentence why they should move up or down.
  4. Propose an updated weight sheet. Only nudge weights (typical delta: 0.02-0.05 per signal). Preserve the sign of `excluded_company_penalty` (negative). Keep `project_match_score` between 0.25 and 0.55. Do NOT introduce new signal names.

Return YAML block-style — no prose, no code fences. Use this exact shape:

overall_assessment: |
  # one paragraph, plain-language summary of what the ranking is getting right and wrong
misplaced_candidates:
  - candidate_id:            # from the input
    current_rank:            # int
    suggested_direction:     # "up" or "down"
    reason: |                # one sentence
updated_weights:             # full weight sheet, same keys as input; nudge each
                             # by a typical delta of 0.02-0.05 in the direction
                             # that would improve the ranking. If the ranking
                             # already looks correct, return the current weights
                             # unchanged (equal to the input). The auditor loop
                             # will stop automatically when your proposals
                             # stabilize across iterations, so returning the
                             # same weights signals "no more useful changes."
  project_match_score:
  recruiter_response_rate_score:
  skill_assessment_score:
  saved_by_recruiters_score:
  tenure_at_product_score:
  github_activity_score_norm:
  interview_completion_score:
  profile_completeness_score_norm:
  open_to_work_score:
  endorsement_rate_score:
  verification_score:
  excluded_company_penalty:
weight_change_rationale: |
  # one paragraph explaining WHY these weight changes should improve the ranking
"""


def _candidate_summary_for_llm(c: dict, rank: int) -> dict:
    """Trim a blended candidate row down to what the LLM actually needs to
    judge fit — no vectors, no raw signal columns the LLM can't interpret."""
    career = []
    try:
        for r in json.loads(c.get("career_json") or "[]"):
            career.append({
                "title": r.get("title"),
                "company": r.get("company"),
                "months": r.get("months"),
                "is_current": r.get("is_current"),
            })
    except json.JSONDecodeError:
        pass

    return {
        "rank": rank,
        "candidate_id": c.get("candidate_id"),
        "final_score": round(c.get("final_score", 0.0), 3),
        "project_match_score": round(c.get("project_match_score", 0.0), 3),
        "current_title": c.get("current_title"),
        "years_of_experience": c.get("years_of_experience"),
        "total_experience_months": c.get("total_experience_months"),
        "city": c.get("city"),
        "willing_to_relocate": c.get("willing_to_relocate"),
        "days_since_active": c.get("days_since_active"),
        "recruiter_response_rate": c.get("recruiter_response_rate"),
        "saved_by_recruiters_30d": c.get("saved_by_recruiters_30d"),
        "interview_completion_rate": c.get("interview_completion_rate"),
        "github_activity_score": c.get("github_activity_score"),
        "profile_completeness_score": c.get("profile_completeness_score"),
        "verified_email": c.get("verified_email"),
        "verified_phone": c.get("verified_phone"),
        "linkedin_connected": c.get("linkedin_connected"),
        "career_history": career,
        # top-1 vector-match citation, to show the LLM what the system saw:
        "top_project_match": _top_evidence(c),
    }


def _top_evidence(c: dict) -> dict | None:
    ev = c.get("per_project_evidence") or []
    if not ev:
        return None
    best = max(ev, key=lambda e: (e.get("similarity") or 0))
    role = best.get("best_role") or {}
    return {
        "project_query": best.get("project"),
        "similarity": best.get("similarity"),
        "matched_role": {
            "company": role.get("company"),
            "title": role.get("title"),
            "recency_label": role.get("recency_label"),
            "description_excerpt": role.get("description_excerpt"),
        } if role else None,
    }


def build_user_prompt(jd_text: str, top_n: list[dict], weights: dict[str, float]) -> str:
    return (
        "JOB DESCRIPTION:\n"
        "-----\n"
        f"{jd_text}\n"
        "-----\n\n"
        f"CURRENT WEIGHTS:\n{yaml.safe_dump(weights, sort_keys=False, default_flow_style=False)}\n"
        f"TOP-{len(top_n)} CANDIDATES (in current ranked order):\n"
        f"{yaml.safe_dump(top_n, sort_keys=False, default_flow_style=False, width=120)}\n\n"
        "Return the YAML judgment now."
    )


# ---------------------------------------------------------------------------
# LLM call.
# ---------------------------------------------------------------------------
def call_llm(jd_text: str, top_n: list[dict], weights: dict[str, float]) -> dict:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("LLM_MODEL_NAME") or os.environ.get("JD_MODEL")
    if not (api_key and base_url and model):
        raise SystemExit(
            "Missing LLM env vars — set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME "
            "(scripts/.env should already do this)."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(jd_text, top_n, weights)},
        ],
    )
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise SystemExit("LLM returned empty content.")
    return _parse_yaml_lenient(content)


# ---------------------------------------------------------------------------
# Pipeline glue.
# ---------------------------------------------------------------------------
def load_jd_text(docx_path: Path) -> str:
    """Best-effort JD text — from docx if available, else compiled JSON summary."""
    if docx_path.exists() and docx_path.suffix.lower() == ".docx":
        try:
            import docx  # python-docx

            doc = docx.Document(str(docx_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            pass
    return docx_path.read_text() if docx_path.exists() else ""


def run_ranker_and_blend(
    jd: dict,
    groups: dict,
    families: list[str],
    db_path: str,
    weights: dict[str, float],
    pool_limit: int,
) -> list[dict]:
    ranked_pool = run_ranking(
        jd=jd,
        groups=groups,
        families=families,
        db_path=db_path,
        k=pool_limit,
        prefilter_limit=pool_limit,
        top_m=200,
        max_inactive_days=180,
        yoe_tolerance_years=1.0,
    )
    if not ranked_pool:
        return []
    jd_skill_bag = list((jd.get("required_technical_skills") or [])) + list(
        (jd.get("preferred_technical_skills") or [])
    )
    jd_exclude_companies = list(jd.get("exclude_companies") or [])
    return weighted_blend(
        ranked_pool,
        weights=weights,
        jd_required_skills=jd_skill_bag,
        jd_exclude_companies=jd_exclude_companies,
        top_k=pool_limit,
    )


def _sanitize_weights(proposed: dict[str, Any], baseline: dict[str, float]) -> dict[str, float]:
    """Force type + guard-rail per-signal. Unknown keys are dropped; missing
    keys inherit the baseline; excluded_company_penalty is force-negative."""
    out: dict[str, float] = {}
    for name, baseline_val in baseline.items():
        v = proposed.get(name, baseline_val)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float(baseline_val)
        if name == "excluded_company_penalty":
            v = -abs(v)
            v = max(v, -0.60)
        elif name == "project_match_score":
            v = max(0.25, min(0.55, v))
        else:
            v = max(0.0, min(0.35, v))
        out[name] = round(v, 4)
    return out


def _write_weights(weights: dict[str, float], path: Path) -> None:
    with path.open("w") as f:
        json.dump(weights, f, indent=2)
        f.write("\n")


def _print_verdict(iteration: int, verdict: dict) -> None:
    print(f"\n=== iteration {iteration} — LLM verdict ===")
    assess = (verdict.get("overall_assessment") or "").strip()
    if assess:
        print(f"\n{assess}")
    misplaced = verdict.get("misplaced_candidates") or []
    if misplaced:
        print(f"\nMisplaced ({len(misplaced)}):")
        for m in misplaced:
            print(
                f"  {m.get('candidate_id')}  rank={m.get('current_rank')}  "
                f"→ {m.get('suggested_direction')}: {m.get('reason','').strip()}"
            )
    rationale = (verdict.get("weight_change_rationale") or "").strip()
    if rationale:
        print(f"\nRationale for weight changes:\n{rationale}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jd-compiled", default="scripts/jd_compiled.json")
    ap.add_argument("--jd-doc", default="India_runs_data_and_ai_challenge/job_description.docx",
                    help="Raw JD (docx or txt) — shown to the LLM verbatim")
    ap.add_argument("--title-groups", default="scripts/title_groups.json")
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--db", default="./redrob_db")
    ap.add_argument("--top-n", type=int, default=20, help="How many candidates the LLM judges")
    ap.add_argument("--pool-limit", type=int, default=2000)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--apply", action="store_true",
                    help="Write LLM-proposed weights to scripts/weights.json each iteration")
    ap.add_argument(
        "--stability-window",
        type=int,
        default=3,
        help="Stop when the last N iterations' proposals are all within --stability-threshold",
    )
    ap.add_argument(
        "--stability-threshold",
        type=float,
        default=0.02,
        help="Max per-signal delta considered 'no change' for stability (default 0.02)",
    )
    args = ap.parse_args()

    jd = json.loads(Path(args.jd_compiled).read_text())
    groups = load_title_groups(args.title_groups)
    jd_text = load_jd_text(Path(args.jd_doc))
    if not jd_text:
        print(f"[warn] Could not read JD from {args.jd_doc}; falling back to compiled JSON dump.", file=sys.stderr)
        jd_text = json.dumps(jd, indent=2)

    weights = load_weights()
    print(f"Starting weights: {json.dumps(weights, indent=2)}")

    # History of weight sheets we've applied — used for the stability check.
    weight_history: list[dict[str, float]] = [dict(weights)]

    for iteration in range(1, args.max_iters + 1):
        # Pipeline needs to see the current weights via the module-level
        # DEFAULT_WEIGHTS that blend() reads if no override is passed. We do
        # pass explicit weights below, so this is defensive.
        import weighted_ranker
        weighted_ranker.DEFAULT_WEIGHTS = weights

        blended = run_ranker_and_blend(
            jd=jd,
            groups=groups,
            families=args.families,
            db_path=args.db,
            weights=weights,
            pool_limit=args.pool_limit,
        )
        if not blended:
            print("Pool is empty — nothing to judge.")
            return 0

        top_n = [_candidate_summary_for_llm(c, i + 1) for i, c in enumerate(blended[:args.top_n])]
        print(f"\nRunning LLM judge on top-{len(top_n)}...")
        verdict = call_llm(jd_text, top_n, weights)

        _print_verdict(iteration, verdict)

        proposed = verdict.get("updated_weights") or {}
        new_weights = _sanitize_weights(proposed, weights)

        # Show a compact diff so it's easy to eyeball.
        any_change = False
        print("\nWeight changes:")
        for k, v in new_weights.items():
            old = weights[k]
            if abs(v - old) > 1e-4:
                print(f"  {k:<34}  {old:+.4f}  →  {v:+.4f}  (Δ {v - old:+.4f})")
                any_change = True
        if not any_change:
            print("  (no weight changes)")

        if args.apply:
            _write_weights(new_weights, WEIGHTS_PATH)
            print(f"\nWrote {WEIGHTS_PATH}")
        else:
            print("\n(--apply not set: weights.json unchanged)")

        weights = new_weights
        weight_history.append(dict(weights))

        # Stability check — look at the last stability_window weight sheets.
        # For every signal, if the max value across the window minus the min
        # is <= stability_threshold, the loop is converging.
        if len(weight_history) >= args.stability_window + 1:
            window = weight_history[-(args.stability_window + 1):]
            max_span = 0.0
            span_signal = None
            for signal in weights:
                values = [w.get(signal, 0.0) for w in window]
                span = max(values) - min(values)
                if span > max_span:
                    max_span = span
                    span_signal = signal
            print(
                f"\n[stability] max signal span over last {args.stability_window} "
                f"iterations: {max_span:.4f} ({span_signal})",
                file=sys.stderr,
            )
            if max_span <= args.stability_threshold:
                print(
                    f"\nWeights stabilized — no signal moved more than "
                    f"{args.stability_threshold} across last {args.stability_window} "
                    f"iterations. Stopping.",
                )
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
