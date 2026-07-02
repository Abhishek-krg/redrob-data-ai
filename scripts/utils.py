"""Shared utilities for the Redrob candidate pipeline.

Anything that both ingestion and ranking need (streaming IO, text normalization,
per-candidate feature extraction) lives here so behavior stays consistent.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterator


# Anchor date for computing days_since_active at ingest time. Kept as a module
# constant so ingestion runs are reproducible and don't drift with wall-clock.
INGEST_TODAY = date(2026, 7, 2)


# ---------------------------------------------------------------------------
# IO — always stream. candidates.jsonl is ~465MB / 2GB parsed; never list().
# ---------------------------------------------------------------------------
def iter_candidates(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield candidate dicts one at a time from a .jsonl or .json file."""
    path = Path(path)
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


# ---------------------------------------------------------------------------
# Text normalization — used to match titles/skills between candidates and JD.
# ---------------------------------------------------------------------------
def normalize_title(title: str | None) -> str:
    return " ".join((title or "").lower().split())


def normalize_skill(name: str | None) -> str:
    """Cheap default. Add a synonym map (js->javascript, k8s->kubernetes, ...)
    when we start seeing false negatives in skill matching."""
    return " ".join((name or "").lower().split())


def city_of(location: str | None) -> str:
    """`profile.location` looks like 'Pune' or 'Bengaluru, Karnataka'.
    Take the first comma-separated segment, lowercased and stripped."""
    if not location:
        return ""
    return location.split(",")[0].strip().lower()


def days_since(date_str: str | None, today: date = INGEST_TODAY) -> int:
    """Whole days between `today` and an ISO date string.
    Returns a large sentinel (10 years) for missing/malformed dates so downstream
    inactivity gates treat "no signal" the same as "very stale"."""
    if not date_str:
        return 3650
    try:
        y, m, d = map(int, date_str.split("-"))
        return (today - date(y, m, d)).days
    except (ValueError, AttributeError):
        return 3650


# ---------------------------------------------------------------------------
# Career-history helpers.
# ---------------------------------------------------------------------------
def most_recent_role(career: list[dict]) -> dict:
    """Return the current role if one is flagged is_current, else the role with
    the latest end_date. Assumes career is non-empty."""
    if not career:
        return {}
    return sorted_career(career)[0]


# ---------------------------------------------------------------------------
# Career-history ordering — used by both the candidate prefilter row (to pick
# most_recent_title) and per-role vector records (to assign role_index).
# ---------------------------------------------------------------------------
def sorted_career(career: list[dict]) -> list[dict]:
    """Sort career_history newest-first.
    Current role wins; ties break on end_date descending.

    The list stored on candidates in candidates.jsonl is NOT guaranteed to be
    in any order, so we normalize once here."""
    def key(r: dict):
        # (is_current: bool as 1/0, end_date_str) — both descending
        return (1 if r.get("is_current") else 0, r.get("end_date") or r.get("start_date") or "")
    return sorted(career, key=key, reverse=True)


def role_text_blob(role: dict) -> str:
    """The exact string embedded per career_history entry. Includes title +
    company so thin-description roles still carry semantic signal."""
    title = (role.get("title") or "").strip()
    company = (role.get("company") or "").strip()
    desc = (role.get("description") or "").strip()
    header = f"{title} at {company}".strip(" at ").strip()
    if header and desc:
        return f"{header}\n{desc}"
    return header or desc


# ---------------------------------------------------------------------------
# Candidate -> flat LanceDB row.
# ---------------------------------------------------------------------------
def candidate_to_record(candidate: dict) -> dict:
    """Flatten one candidate JSON into the columns Candidate schema expects."""
    p = candidate.get("profile") or {}
    s = candidate.get("redrob_signals") or {}
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []

    norm_career = [
        {
            "title": normalize_title(r.get("title")),
            "company": (r.get("company") or "").strip(),
            "months": int(r.get("duration_months") or 0),
            "is_current": bool(r.get("is_current", False)),
            "end_date": r.get("end_date"),
        }
        for r in career
    ]
    norm_skills = [
        {
            "name": normalize_skill(sk.get("name")),
            "proficiency": sk.get("proficiency", "intermediate"),
        }
        for sk in skills
    ]

    mr = most_recent_role(career) if career else {"title": p.get("current_title", "")}

    return {
        "candidate_id": candidate.get("candidate_id"),
        "country": (p.get("country", "") or "").strip(),
        "location": p.get("location", "") or "",
        "city": city_of(p.get("location")),
        "current_title": p.get("current_title", "") or "",
        "most_recent_title": normalize_title(mr.get("title", "")),
        "total_experience_months": sum(r["months"] for r in norm_career),
        "role_titles": sorted({r["title"] for r in norm_career if r["title"]}),
        "skill_names": sorted({sk["name"] for sk in norm_skills if sk["name"]}),
        "years_of_experience": float(p.get("years_of_experience", 0) or 0),
        "open_to_work": bool(s.get("open_to_work_flag", False)),
        "willing_to_relocate": bool(s.get("willing_to_relocate", False)),
        "profile_completeness_score": float(s.get("profile_completeness_score", 0) or 0),
        # -1 means "no GitHub linked" — keep the sentinel intact for the blender.
        "github_activity_score": float(s.get("github_activity_score", 0) if s.get("github_activity_score") is not None else 0),
        "recruiter_response_rate": float(s.get("recruiter_response_rate", 0) or 0),
        "last_active_date": s.get("last_active_date", "") or "",
        "days_since_active": days_since(s.get("last_active_date")),
        "expected_salary_min_lpa": float(
            (s.get("expected_salary_range_inr_lpa") or {}).get("min") or 0
        ),
        "expected_salary_max_lpa": float(
            (s.get("expected_salary_range_inr_lpa") or {}).get("max") or 0
        ),
        "saved_by_recruiters_30d": int(s.get("saved_by_recruiters_30d", 0) or 0),
        "interview_completion_rate": float(s.get("interview_completion_rate", 0) or 0),
        "endorsements_received": int(s.get("endorsements_received", 0) or 0),
        "connection_count": int(s.get("connection_count", 0) or 0),
        "verified_email": bool(s.get("verified_email", False)),
        "verified_phone": bool(s.get("verified_phone", False)),
        "linkedin_connected": bool(s.get("linkedin_connected", False)),
        "skill_assessment_scores_json": json.dumps(s.get("skill_assessment_scores") or {}),
        "career_json": json.dumps(norm_career),
        "skills_json": json.dumps(norm_skills),
    }


# ---------------------------------------------------------------------------
# Per-role rows for the candidate_roles table.
# ---------------------------------------------------------------------------
# Skip roles with description text shorter than this many chars — they'd produce
# noisy embeddings and dilute the top-M vector search results without adding
# real signal. We still keep the candidate; we just don't seed a role vector.
MIN_ROLE_TEXT_CHARS = 20


def iter_candidate_role_records(candidate: dict) -> Iterator[dict]:
    """Yield one CandidateRole row per non-empty career_history entry.

    role_index is 0 for the most recent role, 1 for the next, etc. — this is
    what powers citation labels ("second-most-recent role" == role_index 1).
    """
    career = sorted_career(candidate.get("career_history") or [])
    cid = candidate.get("candidate_id") or ""
    for idx, role in enumerate(career):
        text = role_text_blob(role)
        if len(text) < MIN_ROLE_TEXT_CHARS:
            continue
        yield {
            "role_id": f"{cid}#{idx}",
            "candidate_id": cid,
            "role_index": idx,
            "company": (role.get("company") or "").strip(),
            "title": (role.get("title") or "").strip(),
            "start_date": role.get("start_date") or "",
            "end_date": role.get("end_date") or "",
            "is_current": bool(role.get("is_current", False)),
            "duration_months": int(role.get("duration_months") or 0),
            "text": text,
        }


# ---------------------------------------------------------------------------
# Batching helper for LanceDB inserts and other bulk ops.
# ---------------------------------------------------------------------------
def batched(iterable, size: int):
    """Yield lists of up to `size` items from any iterable."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Title-group helpers — used by the ranking pipeline to translate a set of JD
# target families into the concrete list of candidate titles that qualify.
# ---------------------------------------------------------------------------
def load_title_groups(path: str | Path) -> dict:
    """Load the title_groups.json produced by scripts/title_groups.py."""
    with Path(path).open() as f:
        return json.load(f)


def titles_in_families(
    groups: dict,
    families: list[str],
    include_cross_apply: bool = True,
) -> set[str]:
    """Return the union of titles that belong to any of `families`.

    `groups` is the dict loaded from title_groups.json. `families` is a list
    of family keys (e.g. ['ai_ml_core', 'data_science']). Set
    include_cross_apply=False to keep only titles that sit natively in a family.
    """
    out: set[str] = set()
    fams = groups.get("families", {})
    for fam in families:
        buckets = fams.get(fam, {})
        for entry in buckets.get("primary", []):
            out.add(entry["title"])
        if include_cross_apply:
            for entry in buckets.get("cross_apply", []):
                out.add(entry["title"])
    return out
