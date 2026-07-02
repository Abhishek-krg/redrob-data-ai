# Approach — Redrob India Data & AI Challenge

End-to-end pipeline: raw JD -> compiled JD -> LanceDB ingestion of 100k
candidates -> SQL prefilter -> per-project vector search -> weighted
blend of 12 signals -> XLSX submission. Two evaluation-driven feedback
loops (LLM-as-judge) tune the JD-extraction prompt and the blend
weights.

```
job_description.docx
     |
     v  llm_JD_compiler.py           (Qwen 3.7-plus via Fireworks)
jd_compiled.json  <----- auto_jd_prompt_tuner.py (LLM-as-judge, updates prompt)
     |
     v  ranking.py + weighted_ranker.py
top100_ranked.json <---- auto_rank_adjuster.py (LLM-as-judge, updates weights)
     |
     v  make_submission.py
submission.xlsx (+ submission.csv sanity copy)
```

---

## 1. JD compilation

**Script:** `scripts/llm_JD_compiler.py`
**Prompt file:** `scripts/jd_compiler_prompt.md` (externalised, editable)
**Output:** `scripts/jd_compiled.json`

The raw JD (`.docx`) is converted to text and fed to an LLM that returns
a strict YAML document containing the fields the downstream ranker
actually needs. YAML is used instead of JSON because Qwen 3.7-plus
returned empty responses under `response_format=json_object`; a lenient
YAML parser (`_parse_yaml_lenient`) handles the output.

Fields extracted:

- `min/max_years_experience` — used as an SQL band with ±1 yr tolerance.
- `target_role_titles` — expanded via `title_groups.json` into a title
  IN-list.
- `required_technical_skills`, `preferred_technical_skills` — used for
  the reasoning column, not as a hard gate.
- `expected_projects` — 2-6 SHORT concrete built-systems written in
  engineer-voice. These are the **queries** for the semantic search
  stage, so the compiler prompt specifically forbids JD-voice ("Ship X",
  "Own Y") and meta artefacts ("A/B framework") — those retrieve poorly
  against career descriptions.
- `disqualifiers` — explicit dealbreakers (e.g. pure CV background).
- `location.country / cities / remote_ok / relocation_ok` — the SQL
  location gate. `remote_ok` defaults false unless the JD explicitly
  says fully remote (the prompt catches the common trap of treating
  "hybrid" as remote).
- `visa_sponsorship` — defaults false when the JD is silent.
- `exclude_companies` — list of shops that are a NEGATIVE SIGNAL (TCS,
  Infosys, Wipro, Accenture, Cognizant, Capgemini). This is a
  down-weight list, not a hard filter, and the penalty scales with
  months spent at any of them.

The prompt lives in `jd_compiler_prompt.md` in two H2 sections
(`## SYSTEM_PROMPT`, `## OUTPUT_YAML_TEMPLATE`) so the auto-tuner can
rewrite just the rules without touching the schema.

---

## 2. Ingestion

**Script:** `scripts/ingestion.py`
**Schemas:** `scripts/schemas/candidate.py`
**Store:** LanceDB at `./redrob_db`

### Libraries

- `lancedb` — embedded columnar store with vector search. No server, no
  network, satisfies the "no external APIs at ranking time" contract.
- `sentence-transformers` with `BAAI/bge-base-en-v1.5` (768-dim,
  cosine, normalised). Chosen because bge-base is CPU-inference-cheap
  yet strong on retrieval, and 768 dims keeps the on-disk vector size
  small.
- `torch` with MPS auto-detection on Apple Silicon (~3x throughput over
  CPU) — the encode device is picked at import time by
  `_pick_embedding_device()`; `REDROB_EMBED_DEVICE` overrides.
- `pyarrow` / `pandas` — LanceDB's underlying serialisation.
- `python-docx` — read `.docx` sources.

### Two-table layout

LanceDB is used with a **split schema** — one table for row-level
filters, another for per-role vectors — because the JD scores candidates
based on individual roles, not a concatenated blob.

1. `candidates` (100k rows, NO vector).
   All the columns needed by the SQL prefilter and by the downstream
   blender:
   `candidate_id`, `country`, `city`, `willing_to_relocate`,
   `current_title`, `most_recent_title`, `years_of_experience`,
   `total_experience_months`, `days_since_active`, `open_to_work`,
   `profile_completeness_score`, `github_activity_score`,
   `recruiter_response_rate`, `saved_by_recruiters_30d`,
   `interview_completion_rate`, `endorsements_received`,
   `connection_count`, `verified_email`, `verified_phone`,
   `linkedin_connected`, `skill_assessment_scores_json`,
   `expected_salary_min_lpa`, `expected_salary_max_lpa`, `career_json`
   (whole career history serialised so the blender can compute
   excluded-company tenure and product-company tenure without a
   second lookup), `skill_names`, `role_titles`, `last_active_date`.
2. `candidate_roles` (~350k rows, HAS a 768-dim vector).
   One row per role from `career_history[]` with `role_id =
   f"{candidate_id}#{role_index}"`, `role_index` (0 = most recent),
   `company`, `title`, `start_date`, `end_date`, `duration_months`, a
   truncated `description`, and the bge embedding of the description.
   Descriptions shorter than 20 chars are skipped (they retrieve
   noise).

### Streaming ingest

`candidates.jsonl` is 465 MB. Loading it fully takes ~2 GB of RAM;
streaming line-by-line (`for line in f:`) keeps it near 12 MB. Ingest
batches 512 records at a time into LanceDB, then encodes the
description strings in bge batches of 128 on MPS (auto batch-sized) so
GPU utilisation stays high without spilling.

### No ANN index

The vector index is **flat** — full cosine scan over the pool. After
Stage 1+2 the pool is typically 300–4000 candidates (~1000–15000 role
rows) so a flat scan is faster and more accurate than paying HNSW/IVF
index-build cost, and it makes SQL prefiltering strictly correct
(no missed neighbours). This was a deliberate reversion from an earlier
version that used `top_m=200` per query and silently dropped 98% of the
pool.

---

## 3. First-level ranking

**Script:** `scripts/ranking.py`

Three sequential stages: SQL gate on location, SQL gate on experience
band + title + availability, then a semantic search per JD project.

### Stage 1 — location / visa

SQL predicate on `candidates`:

```
country = jd.location.country
AND (city IN jd.location.cities OR willing_to_relocate = true)
```

Visa is a hard filter only when `jd.visa_sponsorship = false` and the
candidate needs sponsorship.

### Stage 2 — experience band, title family, availability, YoE-consistency

SQL predicate on the survivors:

```
total_experience_months >= (min_years - tol) * 12
AND total_experience_months <= (max_years + tol) * 12
AND current_title IN <expanded from title_groups.json>
AND days_since_active <= 180
AND abs(years_of_experience * 12 - total_experience_months) <= 36
```

The final line is an **inline honeypot filter** — if the candidate
claims 8 years of experience but their career_history only sums to 2
years (or vice versa) they're rejected. The threshold is a ±3-year
delta (a soft version of `honeypot_filter.py`, but bypassed to keep the
gate cheap and inline).

Title family expansion pulls the JD's `target_role_titles` and any
cross-apply groups from `title_groups.json` (default families:
`ai_ml_core`, `data_science`, `backend_engineering`, `data_analytics`).
Computer Vision was explicitly excluded from cross-applies for this JD
because the JD disqualifies CV-primary candidates.

### Stage 3 — per-project semantic search

Only when `position_requirement_type == 'product'` and
`expected_projects` is non-empty.

For each JD project `Q_j`:

1. Embed `Q_j` with bge (same model as ingest, so vectors live in the
   same space).
2. Fetch **all** role vectors for the surviving candidate pool via a
   WHERE clause on `candidate_id IN pool` (no `top_m` cutoff — we
   compute the full NumPy cosine matrix).
3. For each candidate `c`: `best_j[c] = max cosine(Q_j, role)` over the
   candidate's roles. Missing = 0.

Aggregation:

- `project_match_score(c) = mean_j best_j[c]` — average fit across the
  JD's projects (not max), to reward broad relevance over spike matches.
- `per_project_evidence(c)` = the argmax role per project with
  `role_id`, `company`, `title`, `recency_label`, `duration_months`,
  and a description excerpt. Kept end-to-end so the reasoning column
  can cite "built X at Sarvam AI".

Output of first-level ranking is an ordered pool with
`project_match_score` and `per_project_evidence` attached, sorted by
`project_match_score` desc.

---

## 4. Weighted blending

**Script:** `scripts/weighted_ranker.py`
**Weight sheet:** `scripts/weights.json` (externalised; the auto-tuner
writes to this file).

The `project_match_score` alone is insufficient because it only measures
semantic fit against past project descriptions. The blender folds in 11
other signals from `redrob_signals` and the career_history:

| Signal | Meaning | Fallback weight |
| --- | --- | --- |
| `project_match_score` | Stage-3 semantic fit | 0.42 |
| `recruiter_response_rate_score` | Historical response rate | 0.15 |
| `interview_completion_score` | Completion rate | 0.10 |
| `tenure_at_product_score` | Months at product-cos vs services | 0.10 |
| `skill_assessment_score` | Weighted mean of skill_assessment_scores dict | 0.08 |
| `saved_by_recruiters_score` | 30-day saves (min-max normalised) | 0.08 |
| `github_activity_score_norm` | Github signal / 100 (with -1 sentinel handling) | 0.03 |
| `open_to_work_score` | Boolean flag | 0.02 |
| `endorsement_rate_score` | endorsements / connections | 0.02 |
| `profile_completeness_score_norm` | Completeness / 100 | 0.00 |
| `verification_score` | (email + phone + linkedin) / 3 | 0.00 |
| `excluded_company_penalty` | Fraction of career at TCS/Infosys/... | -0.30 |

Each extractor:

- Handles `-1` sentinels explicitly (github_activity_score = -1 means
  "no linked github", not "score of -1").
- Returns a value in `[0, 1]` (the penalty signal returns `[0, 1]` and
  the negative weight applies the sign).
- Never crashes on a missing field.

`blend()` multiplies each signal by its weight, sums, and sorts by
`(-final_score, candidate_id)` to satisfy the challenge's tie-break
rule (validator rejects otherwise). It also attaches a `score_breakdown`
dict per candidate — `{signal: {score, weight, contribution}}` — so
downstream tools (make_submission, auto_rank_adjuster) can inspect
per-signal attribution.

`load_weights()` reads `scripts/weights.json` at import; if the file is
missing or malformed it falls back to `_FALLBACK_WEIGHTS` (identical to
the table above) with a stderr warning.

---

## 5. Evaluation-driven tuning (LLM-as-judge)

Both major artefacts — the JD extraction prompt and the blend weights —
have automated feedback loops that use an LLM as an evaluator, then
apply the LLM's structured suggestions back to the artefact.

### 5a. JD prompt tuner

**Script:** `scripts/auto_jd_prompt_tuner.py`

Loop (per iteration):

1. Read the current `SYSTEM_PROMPT` section from
   `scripts/jd_compiler_prompt.md`.
2. Recompile the raw JD via `llm_JD_compiler.call_llm()` using that
   prompt.
3. Ask a **judge LLM** (same Qwen 3.7-plus, `temperature=0`):
   *"Given the RAW JD, the COMPILED JD, and the CURRENT SYSTEM PROMPT
   — flag anything that was MISSED, MISCLASSIFIED, INVENTED, or
   MISPHRASED, and propose a full replacement SYSTEM_PROMPT."*
4. Judge returns YAML: `done: bool`, `issues_found: [{field, problem,
   fix}]`, `updated_system_prompt: str`, `rationale: str`.
5. If `done: true` — stop. Otherwise, with `--apply`, write the new
   `SYSTEM_PROMPT` section back (preserving the YAML template section)
   and loop.

The `done` field lets the LLM signal "you're already correct" without
being forced to invent problems — this replaced an earlier heuristic
checklist that forced updates every iteration. Section boundaries are
matched with line-anchored regex (`^## SYSTEM_PROMPT[ \t]*$` with
`re.MULTILINE`) to avoid collisions with H2 markers inside the file's
HTML-comment header.

First real run found two genuine issues: the prompt was missing
guidance about disqualifier categories (title-chasers, framework
enthusiasts) and about preferred skills that show up as "bonus" phrasing
(PEFT, open-source contributions).

### 5b. Weight tuner

**Script:** `scripts/auto_rank_adjuster.py`

Loop (per iteration):

1. Load `scripts/weights.json`.
2. Run the full ranking pipeline (`ranking.run()` + `weighted_blend()`)
   to produce the current top-N.
3. Ask a **judge LLM** — payload includes the compiled JD, the current
   weights, and the top-N candidates (with `score_breakdown` and
   `per_project_evidence` inline so the judge can see attribution).
   Prompt: *"Given this JD, would you reorder this top-N? Propose an
   updated weight sheet."*
4. Parse YAML output: `updated_weights: {signal: weight}`,
   `rationale: str`.
5. `_sanitize_weights()` enforces guardrails:
   - `project_match_score` clamped to `[0.25, 0.55]` — the semantic
     match must remain the dominant signal.
   - `excluded_company_penalty` forced negative, clamped to
     `[-0.60, 0.00]`.
   - Other signals clamped to `[0.00, 0.40]`.
6. With `--apply`, write the new sheet to `weights.json` and loop.

**Stopping rule** is numeric, not LLM-driven — an earlier `ranking_ok`
boolean returned by the LLM was noisy and produced premature stops.
Instead the loop keeps a rolling window (default 3 iterations) of
weight vectors and stops when the max per-signal span across the window
is `≤ 0.02`. This gives the LLM room to make small refinements while
guaranteeing convergence in bounded time.

### Why LLM-as-judge over hand-tuned checklists

The first version of the weight tuner was a deterministic checklist
("does the top-N include ≥ N candidates from the target city? does
the mean YoE fall in the band?"). It couldn't distinguish "a mediocre
FAANG candidate at rank 5" from "a strong product-startup candidate at
rank 5" — both passed the checklist. The LLM judge, given the compiled
JD as its rubric, catches those substitution errors because it can
compare candidate narratives against the JD's `expected_projects`,
`disqualifiers`, and `exclude_companies` fields simultaneously.

The two loops share the same pattern (compile -> judge -> structured
suggestion -> apply -> reconverge) but tune orthogonal artefacts, so
they can be run independently in any order.

---

## 6. Submission assembly

**Script:** `scripts/make_submission.py`

- Reads `top100_ranked.json`, re-sorts by `(-final_score,
  candidate_id)` (validator contract), re-emits ranks 1..100.
- Composes reasoning deterministically from each candidate's own record
  (no LLM call, no scores in prose). Depth scales with rank tier:
  - **Tier A (1-25)**: role + company + YoE + top 2 project matches +
    up to 3 JD-matching skills.
  - **Tier B (26-60)**: role + company + YoE + top 1 project match +
    up to 2 JD-matching skills.
  - **Tier C (61-100)**: role + company + YoE + 1 relevant skill.
- Writes `submission.xlsx` (required by the submission portal) plus a
  `submission.csv` copy for `validate_submission.py`. Both files are
  identical in content — the CSV exists only so the provided validator
  can be run locally as a sanity check.

Reproducibility contract (submission_metadata_template.yaml): the full
ranker runs end-to-end from `candidates.jsonl` to `submission.xlsx` in
≤5 minutes on CPU, 16 GB RAM, no network. Ingestion is pre-computed
into `./redrob_db` and declared as an offline artefact.
