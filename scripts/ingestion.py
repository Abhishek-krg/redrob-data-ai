#!/usr/bin/env python3
"""
Ingest candidates.jsonl into two LanceDB tables:

  * `candidates`      — one row per candidate. SQL-prefilter + weighted-ranker
                        signals columns; no vector column. Fast to filter on.
  * `candidate_roles` — one row per career_history entry, with title + company
                        + description embedded to a 768-dim vector. This is
                        where semantic search happens, per project.

This is the SINGLE source of truth for the LanceDB — every column read by
ranking.py, weighted_ranker.py, or any downstream script is written here in
one pass. In particular, on the `candidates` table this ingest emits:

  Prefilter / gate columns:
    candidate_id, country, location, city, current_title, most_recent_title,
    total_experience_months, role_titles, skill_names, years_of_experience,
    open_to_work, willing_to_relocate, days_since_active,
    expected_salary_{min,max}_lpa, last_active_date

  Weighted-ranker signal columns:
    profile_completeness_score, github_activity_score,
    recruiter_response_rate, saved_by_recruiters_30d,
    interview_completion_rate, endorsements_received, connection_count,
    verified_email, verified_phone, linkedin_connected,
    skill_assessment_scores_json

  Payloads (JSON strings) for per-query math:
    career_json  (list of {title, company, months, is_current, end_date})
    skills_json  (list of {name, proficiency})

If you add a new signal to weighted_ranker.py, put its column here in
utils.candidate_to_record() and the schema in schemas/candidate.py — do not
introduce a separate backfill script.

Streams records line-by-line (never loads all 100k candidates into RAM),
flattens each into the shared schemas, and inserts in batches.

Usage:
    python scripts/ingestion.py \\
        --in India_runs_data_and_ai_challenge/candidates.jsonl \\
        --db ./redrob_db

    # Quick smoke test on a subset:
    python scripts/ingestion.py --in ... --sample 5000

    # Skip index build (e.g. staged ingest that will append more rows):
    python scripts/ingestion.py --in ... --no-index

Environment: conda env py3.10 with lancedb, sentence-transformers, tantivy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lancedb

from schemas import Candidate, CandidateRole, EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME
from utils import (
    candidate_to_record,
    iter_candidate_role_records,
    iter_candidates,
)


DEFAULT_DB_PATH = "./redrob_db"
CANDIDATES_TABLE = "candidates"
ROLES_TABLE = "candidate_roles"
DEFAULT_CAND_BATCH = 2000
# Roles carry embeddings — measured sweet spots on this hardware:
#   MPS (bge-base): batch=512 → ~232 rows/s
#   CPU (bge-base): batch=128 → ~77 rows/s
# Bench script at scripts/dev/bench_embeddings.py.
DEFAULT_ROLE_BATCH = 512


# ---------------------------------------------------------------------------
# Table setup.
# ---------------------------------------------------------------------------
def _open_or_create(db, name: str, schema, mode: str):
    """Overwrite when the caller says so or the table doesn't exist yet.
    Otherwise open the existing table for append."""
    if mode == "overwrite" or name not in db.table_names():
        table = db.create_table(name, schema=schema, mode="overwrite")
        print(f"Created table '{name}'.")
    else:
        table = db.open_table(name)
        print(f"Appending to existing table '{name}' ({table.count_rows()} rows).")
    return table


# ---------------------------------------------------------------------------
# One streaming pass — writes to both tables at once so we only read the JSONL
# once and only decode each candidate once.
# ---------------------------------------------------------------------------
def ingest(
    input_path: Path,
    db_path: str,
    cand_batch: int,
    role_batch: int,
    sample: int | None,
    build_indexes: bool,
    mode: str,
) -> None:
    print(
        f"Embedder: {EMBEDDING_MODEL_NAME} on device={EMBEDDING_DEVICE} "
        f"(override with REDROB_EMBED_DEVICE)",
        file=sys.stderr,
    )
    print(
        f"Candidate columns    ({len(Candidate.model_fields)}): "
        f"{sorted(Candidate.model_fields)}",
        file=sys.stderr,
    )
    print(
        f"CandidateRole columns ({len(CandidateRole.model_fields)}): "
        f"{sorted(CandidateRole.model_fields)}",
        file=sys.stderr,
    )
    db = lancedb.connect(db_path)
    candidates_tbl = _open_or_create(db, CANDIDATES_TABLE, Candidate, mode)
    roles_tbl = _open_or_create(db, ROLES_TABLE, CandidateRole, mode)

    cand_buf: list[dict] = []
    role_buf: list[dict] = []
    total_cand = 0
    total_role = 0
    t0 = time.time()

    def flush_cand() -> None:
        nonlocal total_cand
        if cand_buf:
            candidates_tbl.add(cand_buf)
            total_cand += len(cand_buf)
            cand_buf.clear()

    def flush_role() -> None:
        nonlocal total_role
        if role_buf:
            roles_tbl.add(role_buf)
            total_role += len(role_buf)
            role_buf.clear()

    for c in iter_candidates(input_path, limit=sample):
        cand_buf.append(candidate_to_record(c))
        for role in iter_candidate_role_records(c):
            role_buf.append(role)

        if len(cand_buf) >= cand_batch:
            flush_cand()
        if len(role_buf) >= role_batch:
            flush_role()
            elapsed = time.time() - t0
            print(
                f"  {total_cand:>7d} candidates + {total_role:>7d} roles"
                f"  ({total_role / elapsed:.0f} roles/s)",
                file=sys.stderr,
            )

    flush_cand()
    flush_role()

    elapsed = time.time() - t0
    print(
        f"\nIngested {total_cand} candidates + {total_role} roles in {elapsed:.1f}s."
    )

    if build_indexes:
        _build_indexes(candidates_tbl, roles_tbl)

    print(
        f"'{CANDIDATES_TABLE}': {candidates_tbl.count_rows()} rows | "
        f"'{ROLES_TABLE}': {roles_tbl.count_rows()} rows."
    )


# ---------------------------------------------------------------------------
# Index construction.
# ---------------------------------------------------------------------------
def _build_indexes(candidates_tbl, roles_tbl) -> None:
    print("\nBuilding indexes on 'candidates'...")
    t = time.time()
    # BITMAP = tiny cardinality (country) — cheapest possible equality index.
    candidates_tbl.create_scalar_index("country", index_type="BITMAP", replace=True)
    candidates_tbl.create_scalar_index("city", replace=True)
    candidates_tbl.create_scalar_index("location", replace=True)
    candidates_tbl.create_scalar_index("current_title", replace=True)
    candidates_tbl.create_scalar_index("most_recent_title", replace=True)
    candidates_tbl.create_scalar_index("total_experience_months", replace=True)
    candidates_tbl.create_scalar_index("days_since_active", replace=True)
    candidates_tbl.create_scalar_index("role_titles", index_type="LABEL_LIST", replace=True)
    candidates_tbl.create_scalar_index("skill_names", index_type="LABEL_LIST", replace=True)
    print(f"  candidates indexes in {time.time() - t:.1f}s.")

    print("Building indexes on 'candidate_roles'...")
    t = time.time()
    # Scalar index on candidate_id so per-candidate aggregation and
    # WHERE candidate_id IN (...) prefilters at query time are cheap.
    roles_tbl.create_scalar_index("candidate_id", replace=True)
    # FTS lets us also run BM25 hybrid on the role text if we want to later.
    roles_tbl.create_fts_index("text", replace=True)
    print(f"  candidate_roles scalar+FTS indexes in {time.time() - t:.1f}s.")

    # Vector index: NONE. We deliberately want a FLAT (brute-force) index for
    # exact nearest-neighbor search — LanceDB gives us this behavior when no
    # ANN index exists on the vector column. ~400k vectors × 768 dims × 4B is
    # ~1.2 GB scanned per query, which is fine when the query is already
    # prefiltered to a few thousand candidate_ids.
    print("Skipping vector index on candidate_roles.vector — using FLAT (exact) search.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        default="India_runs_data_and_ai_challenge/candidates.jsonl",
        help="Path to candidates.jsonl (or sample .json)",
    )
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="LanceDB directory")
    ap.add_argument("--cand-batch", type=int, default=DEFAULT_CAND_BATCH)
    ap.add_argument("--role-batch", type=int, default=DEFAULT_ROLE_BATCH)
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only ingest first N candidates (for quick smoke tests)",
    )
    ap.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite: replace existing tables. append: add to existing.",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="Skip index build after ingest (both tables)",
    )
    args = ap.parse_args()

    ingest(
        input_path=Path(args.inp),
        db_path=args.db,
        cand_batch=args.cand_batch,
        role_batch=args.role_batch,
        sample=args.sample,
        build_indexes=not args.no_index,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
