#!/usr/bin/env python3
"""Regenerate the README's statistics from artifacts/.

Usage:  python analyze_runs.py [artifacts_dir]

Writes artifacts/analysis/leaderboard_ci.csv and artifacts/analysis/by_type.csv.
Intervals are paired bootstrap percentile intervals over questions
(5,000 resamples, 95%, fixed seed), so numbers are reproducible.
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ART = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
BASELINE = "paragraph_dense_none_raw_0283a68906b3"          # B0
RAW_HYBRID = "recursive_hybrid_none_raw_3f1778529753"       # reference for query transforms
BEST = "recursive_hybrid_none_rerank_bd0feccaf82a"
N_BOOT, SEED = 5000, 0


def load_run(run_id):
    with open(f"{ART}/runs/{run_id}.json") as f:
        return {r["question_id"]: r for r in json.load(f)}


def load_summaries():
    rows = []
    for path in glob.glob(f"{ART}/runs/*_summary.json"):
        with open(path) as f:
            s = json.load(f)
        s["run_id"] = os.path.basename(path)[: -len("_summary.json")]
        rows.append(s)
    return pd.DataFrame(rows).sort_values(["recall_at_k", "mrr_at_k"], ascending=False)


def paired_ci(a, b, metric, qids, idx):
    diff = np.array([a[q][metric] - b[q][metric] for q in qids])
    boots = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def answer_types():
    """Majority annotator answer type per QASPER question (ties -> first annotator)."""
    path = f"{ART}/qasper-dev-v0.3.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for paper in data.values():
        for qa in paper["qas"]:
            types = []
            for a in qa["answers"]:
                ans = a["answer"]
                if ans.get("unanswerable"):
                    types.append("unanswerable")
                elif ans.get("yes_no") is not None:
                    types.append("yes_no")
                elif ans.get("extractive_spans"):
                    types.append("extractive")
                else:
                    types.append("free_form")
            out[qa["question_id"]] = Counter(types).most_common(1)[0][0]
    return out


def main():
    os.makedirs(f"{ART}/analysis", exist_ok=True)
    summ = load_summaries()
    base = load_run(BASELINE)
    raw = load_run(RAW_HYBRID)
    qids = sorted(base)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(qids), (N_BOOT, len(qids)))

    rows = []
    for _, s in summ.iterrows():
        run = load_run(s["run_id"])
        row = {"run_id": s["run_id"], "recall": s["recall_at_k"], "mrr": s["mrr_at_k"],
               "ndcg": s["ndcg_at_k"], "hit": s["hit_at_k"], "latency_s": s["latency_s"],
               "runtime_s": s["runtime_s"]}
        for tag, ref in (("vs_b0", base), ("vs_rawhybrid", raw)):
            for m, name in (("recall_at_k", "recall"), ("mrr_at_k", "mrr")):
                d, lo, hi = paired_ci(run, ref, m, qids, idx)
                row.update({f"d_{name}_{tag}": d, f"d_{name}_{tag}_lo": lo, f"d_{name}_{tag}_hi": hi})
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(f"{ART}/analysis/leaderboard_ci.csv", index=False)

    types = answer_types()
    if types:
        best = load_run(BEST)
        out = []
        for t in ("extractive", "free_form", "yes_no", "unanswerable"):
            ks = [q for q in qids if types.get(q) == t]
            row = {"type": t, "n": len(ks)}
            for m, name in (("recall_at_k", "recall"), ("mrr_at_k", "mrr")):
                d, lo, hi = paired_ci(best, base, m, ks, rng.integers(0, len(ks), (N_BOOT, len(ks))))
                row.update({f"{name}_b0": np.mean([base[q][m] for q in ks]),
                            f"{name}_best": np.mean([best[q][m] for q in ks]),
                            f"d_{name}": d, f"d_{name}_lo": lo, f"d_{name}_hi": hi})
            out.append(row)
        pd.DataFrame(out).round(3).to_csv(f"{ART}/analysis/by_type.csv", index=False)

    print("wrote", f"{ART}/analysis/leaderboard_ci.csv", "and by_type.csv" if types else "")


if __name__ == "__main__":
    main()
