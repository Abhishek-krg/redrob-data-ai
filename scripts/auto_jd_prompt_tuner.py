#!/usr/bin/env python3
"""
LLM-as-judge tuner for the JD compiler's system prompt.


Loop:
  1. Read scripts/jd_compiler_prompt.md (the current system prompt).
  2. Compile the JD via llm_JD_compiler.
  3. Ask a judge LLM: given the RAW JD text, the COMPILED JD, and the
     CURRENT SYSTEM PROMPT — did the compiler miss / mis-extract anything?
     If yes, return an updated SYSTEM_PROMPT section that would fix it.
     If the compiled JD is already consistent with the raw JD and the prompt,
     return `done: true`.
  4. If `done: true`, stop.
  5. Otherwise write the new SYSTEM_PROMPT section back into
     scripts/jd_compiler_prompt.md (preserving the YAML template section),
     recompile, and loop up to --max-iters.

The judge cannot rewrite the OUTPUT_YAML_TEMPLATE — only the rule text
in SYSTEM_PROMPT. That keeps the downstream schema stable while the
extraction guidance evolves.

Usage:
    # Dry-run one iteration, print judgment + proposed prompt (no writes):
    python scripts/auto_jd_prompt_tuner.py

    # Apply proposed prompts and iterate up to 5 times:
    python scripts/auto_jd_prompt_tuner.py --apply --max-iters 5

Env: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME (loaded from scripts/.env).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Preload .env before anything else so LLM_* vars are visible.
from llm_JD_compiler import (  # noqa: E402
    PROMPT_PATH,
    _load_dotenv,
    _parse_yaml_lenient,
    read_jd,
    reload_prompt,
    call_llm as compile_jd_via_llm,
)

_load_dotenv(Path(__file__).with_name(".env"))

import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt for the JUDGE LLM.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are a senior technical recruiter reviewing the output of an LLM-based JD compiler.

You will be given:
  - the RAW JOB DESCRIPTION text (as posted by the employer)
  - the COMPILED JD (a JSON object the compiler produced from the raw JD)
  - the CURRENT SYSTEM PROMPT the compiler used to produce that output

Your job:
  1. Read the raw JD carefully.
  2. Check the compiled JD field-by-field against the raw JD. Flag anything the compiler:
       - MISSED (the JD said it, but the field is null / [] / wrong-typed)
       - MISCLASSIFIED (e.g. put a "required" skill in "preferred", or set remote_ok=true when the JD is hybrid)
       - INVENTED (in the compiled JD but not in the raw text)
       - MISPHRASED (e.g. an expected_project written in meta / JD-voice instead of concrete engineer-voice)
  3. If the compiled JD is faithful to the raw JD and the current prompt gives clear guidance for each rule, return done: true — no prompt update needed.
  4. Otherwise, propose a REPLACEMENT for the SYSTEM PROMPT that would fix the specific issues you found. The replacement must be a full, complete SYSTEM PROMPT, not a diff. Preserve the overall structure (opening paragraph + bulleted rules). Do NOT propose changes to the OUTPUT YAML TEMPLATE — that lives in a separate section of the file and is out of scope.

Return YAML block-style — no prose, no code fences. Use this exact shape:

done:              # bool: true if the compiled JD is already consistent with the raw JD and the prompt is good as-is
issues_found:      # list of concrete problems in the compiled JD; empty if none
  - field:         # e.g. "location.remote_ok"
    problem:       # one sentence
    fix:           # one sentence — how the prompt should change to prevent this
updated_system_prompt: |
  # If done is false, put the FULL replacement SYSTEM PROMPT here.
  # Must be self-contained — the tuner writes this verbatim as the new prompt.
  # If done is true, this may be omitted or set to null.
rationale: |
  # One paragraph explaining why the proposed prompt update should fix the issues.
"""


# ---------------------------------------------------------------------------
# LLM plumbing.
# ---------------------------------------------------------------------------
def _openai_client() -> tuple[OpenAI, str]:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("LLM_MODEL_NAME") or os.environ.get("JD_MODEL")
    if not (api_key and base_url and model):
        raise SystemExit(
            "Missing LLM env vars — set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME "
            "(scripts/.env should already do this)."
        )
    return OpenAI(api_key=api_key, base_url=base_url), model


def call_judge(raw_jd: str, compiled_jd: dict, current_prompt: str) -> dict:
    client, model = _openai_client()
    user_prompt = (
        "RAW JOB DESCRIPTION:\n"
        "-----\n"
        f"{raw_jd}\n"
        "-----\n\n"
        "CURRENT COMPILER SYSTEM PROMPT:\n"
        "-----\n"
        f"{current_prompt}\n"
        "-----\n\n"
        "COMPILED JD (JSON):\n"
        "-----\n"
        f"{json.dumps(compiled_jd, indent=2)}\n"
        "-----\n\n"
        "Return the YAML judgment now."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise SystemExit("Judge LLM returned empty content.")
    return _parse_yaml_lenient(content)


# ---------------------------------------------------------------------------
# File I/O — rewrite ONLY the SYSTEM_PROMPT section in-place, preserving
# everything else (header comment, YAML template section).
# ---------------------------------------------------------------------------
def rewrite_system_prompt_section(path: Path, new_system_prompt: str) -> None:
    text = path.read_text()
    sys_re = re.compile(r"^## SYSTEM_PROMPT[ \t]*$", re.MULTILINE)
    tpl_re = re.compile(r"^## OUTPUT_YAML_TEMPLATE[ \t]*$", re.MULTILINE)
    sys_match = sys_re.search(text)
    tpl_match = tpl_re.search(text)
    if not sys_match or not tpl_match or tpl_match.start() < sys_match.end():
        raise SystemExit(
            f"{path} has malformed section markers — cannot safely rewrite."
        )
    before = text[: sys_match.end()]
    after = text[tpl_match.start() :]
    new_content = before + "\n\n" + new_system_prompt.strip() + "\n\n" + after
    path.write_text(new_content)


def compile_with_current_prompt(input_path: Path) -> dict:
    """Recompile the JD using whatever is currently in jd_compiler_prompt.md.

    We rebind SYSTEM_PROMPT / OUTPUT_YAML_TEMPLATE inside the compiler module
    so its build_user_prompt uses the updated values without a subprocess."""
    import llm_JD_compiler as compiler

    sys_p, tpl_p = reload_prompt()
    compiler.SYSTEM_PROMPT = sys_p
    compiler.OUTPUT_YAML_TEMPLATE = tpl_p

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("LLM_MODEL_NAME") or os.environ.get("JD_MODEL")
    if not (api_key and base_url and model):
        raise SystemExit("Missing LLM env vars — cannot recompile JD.")

    jd_text = read_jd(input_path)
    result = compile_jd_via_llm(
        jd_text=jd_text,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
    )
    # Apply the same safety-net default the compiler CLI applies.
    if result.get("visa_sponsorship") is None:
        result["visa_sponsorship"] = False
    return result


def _print_verdict(iteration: int, verdict: dict) -> None:
    print(f"\n=== iteration {iteration} — judge verdict ===")
    print(f"done: {verdict.get('done')}")
    issues = verdict.get("issues_found") or []
    if issues:
        print(f"\nIssues ({len(issues)}):")
        for i in issues:
            print(
                f"  - field: {i.get('field')}\n"
                f"    problem: {(i.get('problem') or '').strip()}\n"
                f"    fix:     {(i.get('fix') or '').strip()}"
            )
    else:
        print("No issues reported.")
    rationale = (verdict.get("rationale") or "").strip()
    if rationale:
        print(f"\nRationale:\n{rationale}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jd-doc",
        default="India_runs_data_and_ai_challenge/job_description.docx",
        help="Raw JD file (.docx or .txt)",
    )
    ap.add_argument(
        "--compiled-out",
        default="scripts/jd_compiled.json",
        help="Where to write the compiled JD after each iteration (default: overwrite the canonical file)",
    )
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write the proposed prompt back to scripts/jd_compiler_prompt.md each iteration",
    )
    args = ap.parse_args()

    jd_input = Path(args.jd_doc)
    if not jd_input.exists():
        raise SystemExit(f"JD file not found: {jd_input}")
    raw_jd = read_jd(jd_input)

    compiled_out = Path(args.compiled_out)

    for iteration in range(1, args.max_iters + 1):
        # 1. Recompile with the current prompt.
        current_sys_prompt, _ = reload_prompt()
        print(f"\n[iter {iteration}] Compiling JD with current prompt...", file=sys.stderr)
        compiled = compile_with_current_prompt(jd_input)
        compiled_out.parent.mkdir(parents=True, exist_ok=True)
        with compiled_out.open("w") as f:
            json.dump(compiled, f, indent=2)

        # 2. Judge the compiled JD.
        print(f"[iter {iteration}] Running judge on compiled JD...", file=sys.stderr)
        verdict = call_judge(raw_jd, compiled, current_sys_prompt)
        _print_verdict(iteration, verdict)

        if bool(verdict.get("done")):
            print("\nJudge is satisfied. Stopping.")
            break

        proposed = (verdict.get("updated_system_prompt") or "").strip()
        if not proposed:
            print(
                "\nJudge returned done=false but no updated_system_prompt — treating as done.",
                file=sys.stderr,
            )
            break

        if args.apply:
            rewrite_system_prompt_section(PROMPT_PATH, proposed)
            print(f"\nWrote updated SYSTEM_PROMPT to {PROMPT_PATH}")
        else:
            print(
                "\n(--apply not set: jd_compiler_prompt.md unchanged)\n"
                f"Proposed prompt preview (first 400 chars):\n{proposed[:400]}..."
            )
            break  # dry-run: don't loop endlessly with the same prompt

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
