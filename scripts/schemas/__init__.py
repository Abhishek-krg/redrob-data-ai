"""Shared schemas / models / constants used across ingestion, search, and rank."""

from .candidate import (
    Candidate,
    CandidateRole,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    embedder,
)

__all__ = [
    "Candidate",
    "CandidateRole",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DEVICE",
    "embedder",
]
