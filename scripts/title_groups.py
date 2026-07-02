#!/usr/bin/env python3
"""
Group candidate `profile.current_title` values into role families.

Step 2 of the ranking pipeline (target-profile-group identification) needs a
way to say "this candidate's current title is close enough to the target role
to be considered." Titles alone are noisy — a Data Scientist can be a strong
fit for an AI Engineer role, and an ML Engineer working on agents fits an
AI Engineer JD.

This script:
  1. Streams candidates.jsonl and collects unique current_title + counts.
  2. Applies a hand-curated map from role FAMILY -> list of titles that
     credibly cross-apply to that family.
  3. Writes scripts/title_groups.json with the enriched map and per-title
     counts.

Families are keyed by the "hiring intent" — e.g. if the JD is for an
AI Engineer, `ai_engineering` is the pool to draw from.

Families are NOT mutually exclusive: a Data Scientist appears in both
`data_science` (native) and `ai_engineering` (cross-apply). Downstream
scoring should treat the primary family as full credit and cross-apply
families as partial credit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Family definition: primary titles + cross-apply titles.
#
# `primary` = the title sits squarely inside this family. Full match.
# `cross_apply` = the title's holder can credibly move into this family with
#                 minimal ramp-up. Partial match.
#
# Rationale for the split follows Redrob's implied targeting: AI/ML roles are
# the challenge's focus, so those families are drawn tightly; general
# software/data roles cross-apply into AI/ML if the JD is lenient.
FAMILIES: dict[str, dict[str, list[str]]] = {
    # Core AI/ML: research, applied ML, production ML systems.
    "ai_ml_core": {
        "primary": [
            "ML Engineer",
            "Machine Learning Engineer",
            "Senior Machine Learning Engineer",
            "Staff Machine Learning Engineer",
            "Junior ML Engineer",
            "Applied ML Engineer",
            "AI Engineer",
            "Senior AI Engineer",
            "Lead AI Engineer",
            "AI Research Engineer",
            "AI Specialist",
            "Senior Software Engineer (ML)",
            # Computer Vision Engineer intentionally excluded here — the Redrob
            # JD explicitly rejects CV-primary candidates ("re-learning fundamentals
            # here"). If we later run a role that targets CV, add a `cv_core`
            # family rather than mixing CV into ai_ml_core again.
            "NLP Engineer",
            "Senior NLP Engineer",
            "Recommendation Systems Engineer",
            "Search Engineer",
            "Senior Applied Scientist",
        ],
        # People who can credibly move into an AI/ML role with ramp-up.
        "cross_apply": [
            "Data Scientist",
            "Senior Data Scientist",
            "Data Engineer",
            "Senior Data Engineer",
            "Analytics Engineer",
            "Backend Engineer",
            "Software Engineer",
            "Senior Software Engineer",
        ],
    },
    # Data science / analytics: statistical modeling, experimentation, insight.
    "data_science": {
        "primary": [
            "Data Scientist",
            "Senior Data Scientist",
            "Senior Applied Scientist",
            "AI Research Engineer",
        ],
        "cross_apply": [
            "ML Engineer",
            "Machine Learning Engineer",
            "Senior Machine Learning Engineer",
            "Applied ML Engineer",
            "Data Analyst",
            "Analytics Engineer",
            "Business Analyst",
        ],
    },
    # Data engineering / analytics engineering: pipelines, warehouses, dbt.
    "data_engineering": {
        "primary": [
            "Data Engineer",
            "Senior Data Engineer",
            "Analytics Engineer",
        ],
        "cross_apply": [
            "Backend Engineer",
            "Software Engineer",
            "Senior Software Engineer",
            "Cloud Engineer",
            "DevOps Engineer",
            "ML Engineer",
            "Machine Learning Engineer",
        ],
    },
    # Data analysis: dashboards, reporting, SQL-heavy.
    "data_analytics": {
        "primary": [
            "Data Analyst",
            "Business Analyst",
        ],
        "cross_apply": [
            "Analytics Engineer",
            "Data Engineer",
            "Data Scientist",
        ],
    },
    # Backend engineering: services, APIs, distributed systems.
    "backend_engineering": {
        "primary": [
            "Backend Engineer",
            "Software Engineer",
            "Senior Software Engineer",
            "Java Developer",
            ".NET Developer",
        ],
        "cross_apply": [
            "Full Stack Developer",
            "Cloud Engineer",
            "DevOps Engineer",
            "Data Engineer",
            "Senior Data Engineer",
            "ML Engineer",
            "Machine Learning Engineer",
        ],
    },
    # Frontend: UI, web, JS/TS-heavy.
    "frontend_engineering": {
        "primary": [
            "Frontend Engineer",
            "Full Stack Developer",
        ],
        "cross_apply": [
            "Mobile Developer",
            "Software Engineer",
            "Senior Software Engineer",
        ],
    },
    # Full-stack: expected to move across FE + BE.
    "full_stack": {
        "primary": [
            "Full Stack Developer",
            "Software Engineer",
            "Senior Software Engineer",
        ],
        "cross_apply": [
            "Frontend Engineer",
            "Backend Engineer",
            "Java Developer",
            ".NET Developer",
        ],
    },
    # Mobile: iOS/Android.
    "mobile": {
        "primary": [
            "Mobile Developer",
        ],
        "cross_apply": [
            "Frontend Engineer",
            "Full Stack Developer",
            "Software Engineer",
        ],
    },
    # DevOps / SRE / cloud infra.
    "devops_cloud": {
        "primary": [
            "DevOps Engineer",
            "Cloud Engineer",
        ],
        "cross_apply": [
            "Backend Engineer",
            "Software Engineer",
            "Senior Software Engineer",
            "Data Engineer",
            "Senior Data Engineer",
        ],
    },
    # QA / test engineering.
    "qa": {
        "primary": [
            "QA Engineer",
        ],
        "cross_apply": [
            "Software Engineer",
            "Backend Engineer",
            "Full Stack Developer",
        ],
    },
    # Product / project / program management (non-eng).
    "product_project_mgmt": {
        "primary": [
            "Project Manager",
        ],
        "cross_apply": [
            "Operations Manager",
            "Business Analyst",
        ],
    },
    # People / HR.
    "people_hr": {
        "primary": [
            "HR Manager",
        ],
        "cross_apply": [],
    },
    # Sales / GTM.
    "sales_gtm": {
        "primary": [
            "Sales Executive",
        ],
        "cross_apply": [
            "Marketing Manager",
            "Customer Support",
        ],
    },
    # Marketing / content / creative.
    "marketing_creative": {
        "primary": [
            "Marketing Manager",
            "Content Writer",
            "Graphic Designer",
        ],
        "cross_apply": [],
    },
    # Customer-facing support / success.
    "customer_ops": {
        "primary": [
            "Customer Support",
            "Operations Manager",
        ],
        "cross_apply": [
            "Project Manager",
            "Business Analyst",
        ],
    },
    # Finance / accounting.
    "finance": {
        "primary": [
            "Accountant",
        ],
        "cross_apply": [],
    },
    # Non-software engineering disciplines.
    "physical_engineering": {
        "primary": [
            "Mechanical Engineer",
            "Civil Engineer",
        ],
        "cross_apply": [],
    },
}


def collect_titles(candidates_path: Path) -> Counter[str]:
    titles: Counter[str] = Counter()
    with candidates_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            t = (c.get("profile") or {}).get("current_title")
            if t:
                titles[t] += 1
    return titles


def build_output(title_counts: Counter[str]) -> dict:
    all_grouped_titles = set()
    families_out = {}
    for family, buckets in FAMILIES.items():
        primary = buckets["primary"]
        cross = buckets["cross_apply"]
        families_out[family] = {
            "primary": [
                {"title": t, "count": title_counts.get(t, 0)} for t in primary
            ],
            "cross_apply": [
                {"title": t, "count": title_counts.get(t, 0)} for t in cross
            ],
            "primary_total": sum(title_counts.get(t, 0) for t in primary),
            "cross_apply_total": sum(title_counts.get(t, 0) for t in cross),
        }
        all_grouped_titles.update(primary)
        all_grouped_titles.update(cross)

    unmapped = sorted(t for t in title_counts if t not in all_grouped_titles)

    return {
        "families": families_out,
        "unique_titles": [
            {"title": t, "count": n} for t, n in title_counts.most_common()
        ],
        "unmapped_titles": [
            {"title": t, "count": title_counts[t]} for t in unmapped
        ],
        "total_candidates": sum(title_counts.values()),
        "unique_title_count": len(title_counts),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        default="India_runs_data_and_ai_challenge/candidates.jsonl",
    )
    ap.add_argument(
        "--out",
        dest="out",
        default="scripts/title_groups.json",
    )
    args = ap.parse_args()

    counts = collect_titles(Path(args.inp))
    output = build_output(counts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Unique titles: {output['unique_title_count']}")
    print(f"Total candidates with a title: {output['total_candidates']}")
    print(f"Families defined: {len(output['families'])}")
    if output["unmapped_titles"]:
        print(f"Unmapped titles ({len(output['unmapped_titles'])}):")
        for u in output["unmapped_titles"]:
            print(f"  {u['count']:6d}  {u['title']}")
    else:
        print("All titles mapped into at least one family.")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
