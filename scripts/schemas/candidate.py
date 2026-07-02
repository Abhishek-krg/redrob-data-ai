"""LanceDB schemas.

Two tables live side by side in the same LanceDB directory:

  * `candidates`      — one row per candidate. SQL-prefilter columns only
                        (location, YoE, title, availability, salary,
                        + JSON payloads for exact per-query math). No vector.
  * `candidate_roles` — one row per career_history entry. Text + vector for
                        per-project semantic search + traceability metadata
                        (role_index, company, dates) for citation.

Kept in a shared module because ingestion, search, and re-ranking all need
identical column definitions and dimensionality — otherwise they drift.

The embedding model is instantiated once at import time so
`CandidateRole.vector` knows its dimensionality without re-loading the model
per module.
"""

from __future__ import annotations

import os
from typing import List

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"    # 768-dim


def _pick_embedding_device() -> str:
    """Prefer MPS on Apple Silicon (~3x faster than CPU for bge-base at
    batch=512), fall back to CPU everywhere else.

    Override with REDROB_EMBED_DEVICE=cpu (or mps, cuda) to force a specific
    device — useful for benchmarking or if MPS misbehaves for a specific run.
    """
    override = os.environ.get("REDROB_EMBED_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


EMBEDDING_DEVICE = _pick_embedding_device()


embedder = (
    get_registry()
    .get("sentence-transformers")
    .create(name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
)


class Candidate(LanceModel):
    """Prefilter-only row. No embedding — semantic matching lives on the
    candidate_roles table."""
    candidate_id: str

    # --- filter columns (indexed) ---
    country: str
    location: str         # full "City, Region" as given
    city: str             # lowercased first segment — cheap equality gate for location
    current_title: str
    most_recent_title: str
    total_experience_months: int
    role_titles: List[str]
    skill_names: List[str]
    years_of_experience: float
    open_to_work: bool
    willing_to_relocate: bool

    # --- ranking-signal columns (not indexed) ---
    profile_completeness_score: float
    github_activity_score: float           # -1 sentinel = no GitHub linked
    recruiter_response_rate: float
    last_active_date: str
    days_since_active: int                 # today - last_active_date, computed at ingest
    expected_salary_min_lpa: float         # 0 if unknown
    expected_salary_max_lpa: float         # 0 if unknown
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    endorsements_received: int
    connection_count: int
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    # skill_assessment_scores is a dict[skill_name -> 0..100] — stored as a
    # JSON string so LanceDB doesn't need a schema for the (unbounded) keys.
    # Blender parses it back in Python.
    skill_assessment_scores_json: str

    # --- payloads for exact per-query math (e.g. skill_overlap) ---
    career_json: str
    skills_json: str


class CandidateRole(LanceModel):
    """One row per career_history entry. This is where vector search happens.

    role_index is 0 for the most recent role, 1 for the next, etc. — so
    "second-most-recent role" is trivially role_index == 1 for citation.
    """
    role_id: str
    candidate_id: str
    role_index: int
    company: str
    title: str
    start_date: str
    end_date: str
    is_current: bool
    duration_months: int

    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()
