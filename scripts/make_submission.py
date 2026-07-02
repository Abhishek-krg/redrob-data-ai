#!/usr/bin/env python3
"""
Build the final XLSX submission from top100_ranked.json.

Reasoning is composed deterministically from each candidate's own record
(career_json, per_project_evidence, skill_names, years_of_experience) —
no LLM call, no numeric scores in the prose. Depth of the reasoning
scales with rank tier:

  Tier A (rank 1-25)   full: strongest company & role, top 2 project
                       matches, up to 3 JD-matching skills, YoE
  Tier B (rank 26-60)  mid:  strongest company & role, top 1 project
                       match, up to 2 JD-matching skills, YoE
  Tier C (rank 61-100) brief: YoE, one relevant skill, best company

Sort order matches the validator contract: score desc, candidate_id asc
on ties. Ranks re-emitted 1..100 after sort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RANKED = REPO_ROOT / "top100_ranked.json"
DEFAULT_JD = Path(__file__).with_name("jd_compiled.json")


# ---------------------------------------------------------------------------
# Reasoning helpers
# ---------------------------------------------------------------------------
def _load_careers(candidate: dict) -> list[dict]:
    raw = candidate.get("career_json")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []


def _best_projects(candidate: dict, n: int) -> list[dict]:
    evidence = candidate.get("per_project_evidence") or []
    ranked = sorted(evidence, key=lambda e: e.get("similarity", 0.0), reverse=True)
    return ranked[:n]


def _matching_skills(candidate: dict, jd_skills: list[str], limit: int) -> list[str]:
    cand_skills_lower = {s.lower(): s for s in candidate.get("skill_names") or []}
    matched: list[str] = []
    for js in jd_skills:
        key = js.lower()
        if key in cand_skills_lower:
            matched.append(js)  # keep JD casing
            if len(matched) >= limit:
                break
    return matched


def _strongest_company_role(candidate: dict) -> tuple[str, str] | None:
    """Pick the role that best represents the candidate: prefer the role
    tied to the top project match; fall back to current role; then most
    recent."""
    evidence = candidate.get("per_project_evidence") or []
    if evidence:
        top = max(evidence, key=lambda e: e.get("similarity", 0.0))
        br = top.get("best_role") or {}
        company = (br.get("company") or "").strip()
        title = (br.get("title") or "").strip()
        if company and title:
            return title, company

    careers = _load_careers(candidate)
    for role in careers:
        if role.get("is_current"):
            return (role.get("title") or "").strip(), (role.get("company") or "").strip()
    if careers:
        return (careers[0].get("title") or "").strip(), (careers[0].get("company") or "").strip()
    return None


def _yoe_phrase(candidate: dict) -> str:
    yoe = candidate.get("years_of_experience")
    if yoe is None:
        return ""
    return f"{float(yoe):.1f} yrs experience"


def _project_phrase(project_evidence: dict) -> str:
    """Turn a project-evidence dict into 'built X at Company'."""
    br = project_evidence.get("best_role") or {}
    company = (br.get("company") or "").strip()
    proj = (project_evidence.get("project") or "").strip()
    if not proj:
        return ""
    # Trim the project sentence for readability.
    proj = proj.rstrip(".")
    if company:
        return f"built {proj} at {company}"
    return f"built {proj}"


def _companies_visited(candidate: dict, limit: int = 3) -> list[str]:
    seen: list[str] = []
    for role in _load_careers(candidate):
        c = (role.get("company") or "").strip()
        if c and c not in seen:
            seen.append(c)
        if len(seen) >= limit:
            break
    return seen


def compose_reason(candidate: dict, tier: str, jd: dict) -> str:
    jd_skills = list(jd.get("required_technical_skills") or []) + list(
        jd.get("preferred_technical_skills") or []
    )

    yoe = _yoe_phrase(candidate)
    role = _strongest_company_role(candidate)  # (title, company) or None

    if tier == "A":
        skills = _matching_skills(candidate, jd_skills, limit=3)
        projects = _best_projects(candidate, n=2)
        parts: list[str] = []
        if role and role[0] and role[1]:
            parts.append(f"{role[0].title()} at {role[1]}")
        if yoe:
            parts.append(yoe)
        # Project narrative
        proj_phrases = [p for p in (_project_phrase(pe) for pe in projects) if p]
        # Drop the duplicate company mention if we already named it above.
        if role and role[1]:
            company = role[1]
            proj_phrases = [pp.replace(f" at {company}", "", 1) if pp.count(f" at {company}") == 1 else pp for pp in proj_phrases]
        if proj_phrases:
            parts.append("; ".join(proj_phrases))
        if skills:
            parts.append("skills: " + ", ".join(skills))
        return _finalize("; ".join(parts))

    if tier == "B":
        skills = _matching_skills(candidate, jd_skills, limit=2)
        projects = _best_projects(candidate, n=1)
        parts = []
        if role and role[0] and role[1]:
            parts.append(f"{role[0].title()} at {role[1]}")
        if yoe:
            parts.append(yoe)
        proj_phrases = [p for p in (_project_phrase(pe) for pe in projects) if p]
        if role and role[1]:
            company = role[1]
            proj_phrases = [pp.replace(f" at {company}", "", 1) if pp.count(f" at {company}") == 1 else pp for pp in proj_phrases]
        if proj_phrases:
            parts.append(proj_phrases[0])
        if skills:
            parts.append("skills: " + ", ".join(skills))
        return _finalize("; ".join(parts))

    # Tier C
    skills = _matching_skills(candidate, jd_skills, limit=1)
    companies = _companies_visited(candidate, limit=2)
    parts = []
    if role and role[0] and role[1]:
        parts.append(f"{role[0].title()} at {role[1]}")
    elif companies:
        parts.append("prior at " + ", ".join(companies))
    if yoe:
        parts.append(yoe)
    if skills:
        parts.append("relevant skill: " + skills[0])
    return _finalize("; ".join(parts))


def _finalize(text: str) -> str:
    text = text.strip().strip(";").strip()
    if not text:
        return ""
    # Capitalize first letter, add period.
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def _tier_for(rank: int) -> str:
    if rank <= 25:
        return "A"
    if rank <= 60:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_rows(ranked: list[dict], jd: dict) -> list[dict[str, Any]]:
    # Sort per validator contract: score desc, candidate_id asc on ties.
    ordered = sorted(
        ranked, key=lambda c: (-float(c["final_score"]), c["candidate_id"])
    )
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(ordered, start=1):
        tier = _tier_for(i)
        reason = compose_reason(cand, tier, jd)
        rows.append(
            {
                "candidate_id": cand["candidate_id"],
                "rank": i,
                "score": round(float(cand["final_score"]), 6),
                "reasoning": reason,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranked", default=str(DEFAULT_RANKED), help="top100_ranked.json path")
    ap.add_argument("--jd", default=str(DEFAULT_JD), help="jd_compiled.json path")
    ap.add_argument(
        "--out",
        required=True,
        help="Output XLSX path (e.g. team_xxx.xlsx). A sibling .csv is also emitted for validate_submission.py.",
    )
    args = ap.parse_args()

    ranked = json.loads(Path(args.ranked).read_text())
    jd = json.loads(Path(args.jd).read_text())

    if len(ranked) < 100:
        raise SystemExit(f"Expected at least 100 ranked candidates, got {len(ranked)}")
    ranked = ranked[:100]

    rows = build_rows(ranked, jd)
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

    out_xlsx = Path(args.out)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"Wrote {out_xlsx} ({len(df)} rows)")

    out_csv = out_xlsx.with_suffix(".csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (for validate_submission.py)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
