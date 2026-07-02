# redrob-data-ai

Submission for the **Redrob India Data & AI Challenge** — rank the top 100
candidates from 100,000 profiles for a Senior AI Engineer JD.

Full design writeup is in [`approach.md`](approach.md) and the
architecture diagram is at [`system_architecture.png`](system_architecture.png).

---

## What this repo contains

```
scripts/
  ingestion.py             # streaming JSONL -> LanceDB (two tables)
  ranking.py               # SQL prefilter + per-project vector search
  weighted_ranker.py       # 12-signal weighted blend
  utils.py                 # shared helpers (title expansion, record shaping)
  schemas/                 # LanceDB Candidate + CandidateRole schemas
  title_groups.py          # offline: builds title_groups.json
  honeypot_filter.py       # YoE-consistency reference filter
  llm_JD_compiler.py       # raw JD (.docx) -> jd_compiled.json (YAML/JSON)
  auto_jd_prompt_tuner.py  # LLM-as-judge tuner for the JD extraction prompt
  auto_rank_adjuster.py    # LLM-as-judge tuner for the blend weights
  make_submission.py       # top100_ranked.json -> submission.xlsx (+ .csv)

  # Config / prompt / artefacts (edited by tuners or humans)
  weights.json             # 12-signal weight sheet
  title_groups.json        # canonical title family -> role list
  jd_compiler_prompt.md    # editable rules for the JD compiler
  jd_compiled.json         # compiled JD (checked in for reproducibility)

validate_submission.py         # organiser-provided CSV validator
submission_metadata_template.yaml
approach.md
system_architecture.png
```

Data files (`candidates.jsonl`, `job_description.docx`, `redrob_db/`,
`top100_ranked.json`, `submission.xlsx`) are **not** shipped in this repo.
Place them as described below.

---

## Setup

### 1. Python environment

Python 3.10 recommended (this is what we developed and tested against).

```bash
conda create -n redrob python=3.10 -y
conda activate redrob
```

Or with `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Dependencies

```bash
pip install \
  lancedb \
  sentence-transformers \
  torch \
  numpy \
  pandas \
  openpyxl \
  pyyaml \
  python-docx \
  openai \
  python-dotenv \
  pyarrow
```

`torch` will grab a CPU wheel by default. On Apple Silicon, the ingest
script auto-detects MPS for ~3x throughput; no extra install needed.

### 3. Environment variables (only for offline LLM steps)

Ranking itself uses **no external APIs**. LLM access is only required
for the offline JD compilation and the two tuner loops. Create
`scripts/.env`:

```
LLM_API_KEY=<your fireworks key>
LLM_BASE_URL=https://api.fireworks.ai/inference/v1
LLM_MODEL_NAME=accounts/fireworks/models/qwen3-235b-a22b-instruct-2507
```

Any OpenAI-compatible endpoint will work.

### 4. Data layout

Place the organiser-provided data alongside this repo:

```
<parent>/
  redrob-data-ai/                     # this repo
  India_runs_data_and_ai_challenge/
    candidates.jsonl                  # 100k profiles, 465 MB
    job_description.docx
    candidate_schema.json
    validate_submission.py
    ...
```

---

## Run the pipeline

All commands below assume `cwd` is the repo root.

### Step 1 — Compile the JD (once per JD, ~10s)

```bash
python scripts/llm_JD_compiler.py \
    --jd ../India_runs_data_and_ai_challenge/job_description.docx \
    --out scripts/jd_compiled.json
```

A pre-compiled `jd_compiled.json` is checked in — you can skip this
step for the default JD.

Optional: run the LLM-as-judge prompt tuner to refine the extraction
rules:

```bash
python scripts/auto_jd_prompt_tuner.py --apply --max-iters 5
```

### Step 2 — Ingest candidates into LanceDB (one-time, ~15–25 min on MPS, ~45 min CPU)

```bash
python scripts/ingestion.py \
    --input ../India_runs_data_and_ai_challenge/candidates.jsonl \
    --db ./redrob_db
```

This builds two tables — `candidates` (metadata + 12 signals, no vector)
and `candidate_roles` (one row per past role with a 768-dim bge
embedding). Idempotent; safe to re-run.

Force CPU / MPS explicitly:

```bash
REDROB_EMBED_DEVICE=cpu python scripts/ingestion.py ...
REDROB_EMBED_DEVICE=mps python scripts/ingestion.py ...
```

### Step 3 — Rank + blend

```bash
python scripts/ranking.py \
    --jd scripts/jd_compiled.json \
    --title-groups scripts/title_groups.json \
    --db ./redrob_db \
    --k 500 \
    --top-k-blended 100 \
    --blended-out top100_ranked.json
```

Outputs `top100_ranked.json` — 100 candidates with `final_score`,
`score_breakdown`, and `per_project_evidence`. Runs in <1 min on the
pre-filtered pool.

Optional: run the LLM-as-judge weight tuner to refine `weights.json`:

```bash
python scripts/auto_rank_adjuster.py \
    --apply --max-iters 5 \
    --stability-window 3 --stability-threshold 0.02
```

Rerun Step 3 after each `--apply` iteration.

### Step 4 — Build the submission

```bash
python scripts/make_submission.py --out submission.xlsx
```

Emits:
- `submission.xlsx` — required by the submission portal.
- `submission.csv` — identical content, for local validation.

### Step 5 — Validate

```bash
python validate_submission.py submission.csv
```

Expected: `Submission is valid.`

---

## Reproducibility

Per `submission_metadata_template.yaml`, the ranker runs end-to-end in
**≤5 minutes** on CPU / 16 GB RAM / no network, once ingestion is
pre-built. Ingestion is the one long step; it's a declared offline
artefact (`./redrob_db`).

The pipeline is deterministic:
- Embedding model + dim: `BAAI/bge-base-en-v1.5`, 768-d cosine, normalised.
- Vector search is **flat** (full cosine over the prefiltered pool) —
  no ANN cutoff, so results don't depend on index build order.
- Blend weights are pinned in `weights.json` and version-controlled.
- Tie-break is `candidate_id ↑` per the challenge spec.

---

## One-liner (end-to-end after ingestion)

```bash
python scripts/ranking.py --jd scripts/jd_compiled.json \
    --title-groups scripts/title_groups.json --db ./redrob_db \
    --k 500 --top-k-blended 100 --blended-out top100_ranked.json \
&& python scripts/make_submission.py --out submission.xlsx \
&& python validate_submission.py submission.csv
```
