#!/usr/bin/env python3
"""
Redrob first-level ranking.

Pipeline:

  Stage 1  Location / relocation / visa gate           -> SQL on `candidates`
  Stage 2  Experience band + title-family + availability -> SQL on `candidates`
             experience band uses jd.min_years_experience & max_years_experience
             with a ±1y tolerance (override with --yoe-tolerance).
  Stage 3  Per-project semantic search + aggregation   -> vector on `candidate_roles`

Stage 3 (only when position_requirement_type == 'product'):
  * Each JD expected_project is embedded as a SEPARATE query vector.
  * For each project j we run vector search on candidate_roles WHERE
    candidate_id IN <prefiltered pool> and take top-M role hits.
  * Per candidate c: best_j[c] = max cosine similarity between any of c's
    roles and query Q_j. Missing hit -> 0.
  * score(c) = mean(best_j[c])  — average fit across the JD's projects.
  * evidence(c) = the argmax role per project (role_id, company, title,
    dates, role_index, similarity, description excerpt) so the citation can
    say "second-most-recent role at Acme (Senior ML Engineer, 2022-2024)".

This module does FIRST-LEVEL RANKING ONLY — vector-search based match on
projects. It does not blend in profile_completeness, github_activity,
recruiter_response_rate, or skill_overlap. Those are the downstream
blender's responsibility and take (candidate_id, score, evidence) as input.

Usage:
    python scripts/ranking.py \\
        --jd scripts/jd_compiled.json \\
        --title-groups scripts/title_groups.json \\
        --db ./redrob_db \\
        --k 100 --top-m 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lancedb
import numpy as np

from schemas import embedder
from utils import load_title_groups, titles_in_families
from weighted_ranker import blend as weighted_blend


DEFAULT_FAMILIES = ["ai_ml_core", "data_science", "backend_engineering", "data_analytics"]
CANDIDATES_TABLE = "candidates"
ROLES_TABLE = "candidate_roles"

# LanceDB returns a similarity score under a slightly different column name
# depending on version — accept any of these.
_SCORE_KEYS = ("_relevance_score", "_score", "_distance")


def _score(row: dict) -> float:
    """Return a cosine similarity in [0, 1] regardless of what LanceDB reports.

    LanceDB returns *distance* (1 - cosine_sim) since our vectors are
    normalized by bge — convert to similarity by (1 - distance). If a version
    surfaces an explicit `_relevance_score` / `_score` we honor that first.
    """
    for k in ("_relevance_score", "_score"):
        v = row.get(k)
        if v is not None:
            return float(v)
    d = row.get("_distance")
    if d is not None:
        return 1.0 - float(d)
    return 0.0


# ---------------------------------------------------------------------------
# SQL predicate builders (unchanged from previous step).
# ---------------------------------------------------------------------------
def _sql_str(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def _sql_in(values) -> str:
    return "(" + ", ".join(_sql_str(v) for v in values) + ")"


def build_location_predicate(jd_location: dict, visa_sponsorship: bool) -> list[str]:
    remote_ok = bool(jd_location.get("remote_ok"))
    country = (jd_location.get("country") or "").strip()
    cities = [c.strip().lower() for c in (jd_location.get("cities") or []) if c]

    clauses: list[str] = []
    if remote_ok:
        return clauses

    if not visa_sponsorship and country:
        clauses.append(f"country = {_sql_str(country)}")

    if cities:
        clauses.append(
            f"(city IN {_sql_in(cities)} OR willing_to_relocate = true)"
        )
    else:
        clauses.append("willing_to_relocate = true")
    return clauses


def build_experience_title_predicate(
    min_years: float | None,
    max_years: float | None,
    allowed_titles: set[str],
    yoe_tolerance_years: float = 1.0,
) -> list[str]:
    """SQL predicates for the experience + title gate.

    min/max experience come from the compiled JD (JD's stated band). We apply
    a ± tolerance (default ±1 year) on BOTH ends because the JD's band is
    aspirational rather than hard — a "5-9 years" role should not reject a
    strong 4.5y or 9.5y candidate on a technicality. Turn tolerance off with
    yoe_tolerance_years=0.
    """
    clauses: list[str] = []
    tol_months = int(float(yoe_tolerance_years) * 12)
    if min_years is not None:
        lower_months = max(0, int(float(min_years) * 12) - tol_months)
        clauses.append(f"total_experience_months >= {lower_months}")
    if max_years is not None:
        upper_months = int(float(max_years) * 12) + tol_months
        clauses.append(f"total_experience_months <= {upper_months}")
    if allowed_titles:
        clauses.append(f"current_title IN {_sql_in(sorted(allowed_titles))}")
    return clauses


def build_availability_predicate(max_inactive_days: int) -> list[str]:
    return [f"days_since_active <= {int(max_inactive_days)}"]


# Impossible-YoE gate: candidates whose self-declared years_of_experience
# disagrees with the sum of their career_history durations by more than
# YOE_MISMATCH_MAX_MONTHS are internally-inconsistent and dropped. Keeps
# the filter self-contained here — no dependency on scripts/honeypot_filter.py.
YOE_MISMATCH_MAX_MONTHS = 36   # 3 years


def build_yoe_consistency_predicate() -> list[str]:
    return [
        f"abs(years_of_experience * 12 - total_experience_months) <= {YOE_MISMATCH_MAX_MONTHS}"
    ]


# ---------------------------------------------------------------------------
# Stage 1 + 2: SQL prefilter over the candidates table.
# ---------------------------------------------------------------------------
def prefilter_candidates(
    db,
    jd: dict,
    groups: dict,
    families: list[str],
    max_inactive_days: int,
    prefilter_limit: int,
    yoe_tolerance_years: float = 1.0,
) -> list[dict]:
    location = jd.get("location") or {}
    visa = bool(jd.get("visa_sponsorship", False))
    allowed_titles = titles_in_families(groups, families, include_cross_apply=True)

    clauses = (
        build_location_predicate(location, visa)
        + build_experience_title_predicate(
            jd.get("min_years_experience"),
            jd.get("max_years_experience"),
            allowed_titles,
            yoe_tolerance_years=yoe_tolerance_years,
        )
        + build_availability_predicate(max_inactive_days)
        + build_yoe_consistency_predicate()
    )
    where = " AND ".join(clauses) if clauses else None
    print(f"[stage 1+2] SQL prefilter: {where or '(none)'}", file=sys.stderr)

    cand_tbl = db.open_table(CANDIDATES_TABLE)
    q = cand_tbl.search()
    if where:
        q = q.where(where)
    return q.limit(prefilter_limit).to_list()


# ---------------------------------------------------------------------------
# Stage 3: per-project vector search + aggregation.
# ---------------------------------------------------------------------------
def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th', ..."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _recency_label(role_index: int, is_current: bool) -> str:
    if role_index == 0:
        return "current role" if is_current else "most recent role"
    if role_index == 1:
        return "second-most-recent role"
    return f"{_ordinal(role_index + 1)}-most-recent role"


def _fetch_pool_roles(db, candidate_ids: list[str], chunk_size: int = 5000) -> list[dict]:
    """Return ALL role rows for the candidates in the prefiltered pool.
    Chunked so a single WHERE IN (...) never blows past LanceDB's clause size."""
    roles_tbl = db.open_table(ROLES_TABLE)
    rows: list[dict] = []
    for start in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[start : start + chunk_size]
        where = f"candidate_id IN {_sql_in(chunk)}"
        rows.extend(roles_tbl.search().where(where).limit(len(chunk) * 20).to_list())
    return rows


def per_project_search(
    db,
    projects: list[str],
    candidate_ids: list[str],
    top_m: int,  # kept for CLI compatibility; unused in this implementation
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, dict]]]:
    """Score every role in the pool against every project query, then keep the
    best-matching role per (candidate, project).

    We fetch all pool roles once, then compute cosine similarity in NumPy
    against the N project vectors — exact, covers every candidate, and skips
    the "top-M cutoff misses most of the pool" trap of using .search().limit().

    Returns:
        best_sim[candidate_id][project_index] = max cosine sim across c's roles
        best_role[candidate_id][project_index] = the role row that produced it
    """
    pool_roles = _fetch_pool_roles(db, candidate_ids)
    print(
        f"[stage 3] fetched {len(pool_roles)} pool roles for scoring",
        file=sys.stderr,
    )

    if not pool_roles:
        return defaultdict(dict), defaultdict(dict)

    # (n_roles, dim) matrix of role vectors, already unit-normalized by bge.
    role_vecs = np.asarray(
        [r["vector"] for r in pool_roles], dtype=np.float32
    )
    # Renormalize defensively in case any row got denormalized in transit.
    row_norms = np.linalg.norm(role_vecs, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    role_vecs = role_vecs / row_norms

    # (n_projects, dim) matrix of query vectors.
    query_vecs = np.asarray(
        embedder.generate_embeddings(projects), dtype=np.float32
    )
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_norms[query_norms == 0] = 1.0
    query_vecs = query_vecs / query_norms

    # (n_projects, n_roles) similarity matrix.
    sim_matrix = query_vecs @ role_vecs.T

    best_sim: dict[str, dict[int, float]] = defaultdict(dict)
    best_role: dict[str, dict[int, dict]] = defaultdict(dict)

    for j, project_text in enumerate(projects):
        sims_j = sim_matrix[j]
        for i, role in enumerate(pool_roles):
            cid = role.get("candidate_id")
            if not cid:
                continue
            sim = float(sims_j[i])
            prev = best_sim[cid].get(j)
            if (
                prev is None
                or sim > prev
                or (
                    sim == prev
                    and role.get("role_index", 999)
                    < best_role[cid][j].get("role_index", 999)
                )
            ):
                best_sim[cid][j] = sim
                best_role[cid][j] = role

    return best_sim, best_role


def build_evidence(
    projects: list[str],
    best_sim: dict[str, dict[int, float]],
    best_role: dict[str, dict[int, dict]],
    candidate_ids: list[str],
    excerpt_chars: int = 220,
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """Turn per-project bests into (score, evidence) per candidate.

    score(c) = mean(best_sim[c][j]) over ALL j in 0..N-1 (missing -> 0).
    evidence(c) = list ordered by project index, one entry per project,
                  each containing the argmax role and its similarity.
    """
    scores: dict[str, float] = {}
    evidence: dict[str, list[dict]] = {}
    n = len(projects)

    for cid in candidate_ids:
        per_project = []
        sum_sim = 0.0
        for j, project in enumerate(projects):
            role = best_role.get(cid, {}).get(j)
            sim = best_sim.get(cid, {}).get(j, 0.0)
            sum_sim += max(sim, 0.0)
            if role is None:
                per_project.append(
                    {
                        "project": project,
                        "similarity": 0.0,
                        "best_role": None,
                    }
                )
            else:
                desc = role.get("text") or ""
                # role.text is "Title at Company\ndescription" — cite the
                # description portion for excerpting.
                excerpt = desc.split("\n", 1)[-1].strip()
                if len(excerpt) > excerpt_chars:
                    excerpt = excerpt[:excerpt_chars].rsplit(" ", 1)[0] + "…"
                per_project.append(
                    {
                        "project": project,
                        "similarity": round(float(sim), 4),
                        "best_role": {
                            "role_id": role.get("role_id"),
                            "role_index": role.get("role_index"),
                            "recency_label": _recency_label(
                                int(role.get("role_index", 0)),
                                bool(role.get("is_current", False)),
                            ),
                            "company": role.get("company"),
                            "title": role.get("title"),
                            "start_date": role.get("start_date"),
                            "end_date": role.get("end_date"),
                            "duration_months": role.get("duration_months"),
                            "description_excerpt": excerpt,
                        },
                    }
                )
        scores[cid] = round(sum_sim / n, 4) if n else 0.0
        evidence[cid] = per_project

    return scores, evidence


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------
def run(
    jd: dict,
    groups: dict,
    families: list[str],
    db_path: str,
    k: int,
    prefilter_limit: int,
    top_m: int,
    max_inactive_days: int = 180,
    yoe_tolerance_years: float = 1.0,
) -> list[dict]:
    db = lancedb.connect(db_path)

    prefiltered = prefilter_candidates(
        db,
        jd,
        groups,
        families,
        max_inactive_days,
        prefilter_limit,
        yoe_tolerance_years=yoe_tolerance_years,
    )
    print(f"[stage 1+2] pool after SQL prefilter: {len(prefiltered)}", file=sys.stderr)
    if not prefiltered:
        return []

    # Extract identifiers we need for stage 3 + evidence enrichment.
    by_id = {c["candidate_id"]: c for c in prefiltered}
    candidate_ids = list(by_id.keys())

    projects = _projects_from_jd(jd)
    if not projects:
        print(
            "[stage 3] no expected_projects / not a 'product' role — scoring is 0.0 "
            "for all candidates; downstream blender should own ranking.",
            file=sys.stderr,
        )
        results = [
            _assemble(cid, by_id[cid], 0.0, []) for cid in candidate_ids[:k]
        ]
        return results

    print(f"[stage 3] {len(projects)} project queries × {len(candidate_ids)} candidates", file=sys.stderr)
    best_sim, best_role = per_project_search(db, projects, candidate_ids, top_m)
    scores, evidence = build_evidence(projects, best_sim, best_role, candidate_ids)

    ranked_ids = sorted(candidate_ids, key=lambda cid: scores.get(cid, 0.0), reverse=True)
    ranked_ids = ranked_ids[:k]

    return [_assemble(cid, by_id[cid], scores[cid], evidence[cid]) for cid in ranked_ids]


def _projects_from_jd(jd: dict) -> list[str]:
    if (jd.get("position_requirement_type") or "").lower() != "product":
        return []
    projects = jd.get("expected_projects") or []
    return [p.strip() for p in projects if isinstance(p, str) and p.strip()]


def _assemble(candidate_id: str, candidate_row: dict, score: float, evidence: list[dict]) -> dict:
    """Trim the LanceDB row of heavy columns and attach the score + evidence.

    NOTE: `career_json` is kept because the downstream weighted_ranker signals
    (tenure_at_product_score, excluded_company_penalty) need per-role
    company/months from it. `skills_json` is dropped because the skill
    signal reads from `skill_assessment_scores_json` instead.
    """
    row = {k: v for k, v in candidate_row.items() if k not in ("skills_json",)}
    row["project_match_score"] = score
    row["per_project_evidence"] = evidence
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jd", default="scripts/jd_compiled.json", help="Compiled JD JSON")
    ap.add_argument("--title-groups", default="scripts/title_groups.json")
    ap.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
        help="Target families from title_groups.json (space-separated)",
    )
    ap.add_argument("--db", default="./redrob_db")
    ap.add_argument("--k", type=int, default=100, help="Number of ranked results to return")
    ap.add_argument(
        "--prefilter-limit",
        type=int,
        default=20000,
        help="Cap on the SQL-prefiltered pool going into stage 3 (default 20k)",
    )
    ap.add_argument(
        "--top-m",
        type=int,
        default=200,
        help="Top-M roles retrieved per project query per candidate-id chunk (default 200)",
    )
    ap.add_argument(
        "--max-inactive-days",
        type=int,
        default=180,
        help="Drop candidates inactive for more than this many days (default 180)",
    )
    ap.add_argument(
        "--yoe-tolerance",
        type=float,
        default=1.0,
        help="± years of tolerance applied to the JD's YoE band (default 1.0)",
    )
    ap.add_argument("--out", default=None, help="Optional JSON path to write the raw ranker shortlist to")
    ap.add_argument(
        "--blended-out",
        default="top100_ranked.json",
        help="Path for the final weighted-ranker output (default top100_ranked.json)",
    )
    ap.add_argument(
        "--top-k-blended",
        type=int,
        default=100,
        help="How many candidates to keep after the weighted blend (default 100)",
    )
    args = ap.parse_args()

    with Path(args.jd).open() as f:
        jd = json.load(f)
    groups = load_title_groups(args.title_groups)

    results = run(
        jd=jd,
        groups=groups,
        families=args.families,
        db_path=args.db,
        k=args.k,
        prefilter_limit=args.prefilter_limit,
        top_m=args.top_m,
        max_inactive_days=args.max_inactive_days,
        yoe_tolerance_years=args.yoe_tolerance,
    )

    print(f"\nReturned {len(results)} candidates (by project_match_score desc).")
    for r in results[:20]:
        cid = r.get("candidate_id")
        best = r.get("per_project_evidence") or []
        top_role = None
        if best:
            # Show the highest-similarity project for a compact preview.
            top = max(best, key=lambda x: x.get("similarity") or 0.0)
            role = top.get("best_role")
            if role:
                top_role = f"{role.get('recency_label')} @ {role.get('company')} ({role.get('title')})"
        print(
            f"  {cid}  score={r.get('project_match_score')}  "
            f"title={r.get('current_title')!r}  city={r.get('city')!r}  "
            f"top_match={top_role}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote raw ranker output: {out_path}")

    # ---- Weighted blend on top of the first-level ranker output ----
    # Feed both required + preferred to the assessment-signal extractor. The
    # required list is often paraphrased ("embeddings-based retrieval") and
    # won't match candidates' canonical token names; preferred fills that gap.
    jd_skill_bag = list(jd.get("required_technical_skills") or []) + list(
        jd.get("preferred_technical_skills") or []
    )
    jd_exclude_companies = list(jd.get("exclude_companies") or [])
    blended = weighted_blend(
        results,
        jd_required_skills=jd_skill_bag,
        jd_exclude_companies=jd_exclude_companies,
        top_k=args.top_k_blended,
    )
    blended_out = Path(args.blended_out)
    blended_out.parent.mkdir(parents=True, exist_ok=True)
    with blended_out.open("w") as f:
        json.dump(blended, f, indent=2)
    print(f"Wrote blended top-{args.top_k_blended}: {blended_out}")
    print(f"\nTop 10 after weighted blend (final_score desc):")
    for r in blended[:10]:
        print(
            f"  #{r['rank']}  {r['candidate_id']}  "
            f"final={r['final_score']}  "
            f"proj={r.get('project_match_score')}  "
            f"resp={r.get('recruiter_response_rate_score')}  "
            f"skill={r.get('skill_assessment_score')}  "
            f"gh={r.get('github_activity_score_norm')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
