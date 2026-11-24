# RAGxiv — Retrieval & Answer Quality Experiments

> **Reproducible results snapshot.** Every number below is computed from the exported `artifacts/` directory by [`analyze_runs.py`](analyze_runs.py) (retrieval statistics) or read directly from `generation_scores.csv` and `ragas_scores.csv` (answer quality). Results describe this one run of the QASPER benchmark with small local models; they are not general rankings of RAG methods.

## TL;DR

1. **No retrieval method reliably beats a plain paragraph-chunk dense baseline on this sample.** The nominal winner (recursive chunks + hybrid BM25/dense + cross-encoder rerank) gains +0.013 Recall@5, but its 95% paired-bootstrap interval is [−0.040, +0.065], which includes zero.
2. **Only two effects are statistically reliable, and both are negative:** BM25-only retrieval (about −0.09 Recall@5) and sentence-window indexing (about −0.06).
3. **LLM query transformation (rewrite, multi-query, HyDE) never helped** and cost 37× to 734× more latency per query. None of the drops is individually significant, but all point the same way.
4. **Retrieval is the ceiling for answer quality.** Across the 30 generated answers, all 11 correct answers and all 7 partially correct answers had a gold paragraph in the top-5; 8 of the 12 wrong or abstained answers had none. The other 4 failed even with some gold evidence retrieved.

## At a glance

| Measure | Observed |
| ------- | -------- |
| Completed retrieval configurations | **25** (no failed sweeps; `sweep_failures.json` is empty) |
| QASPER questions in the retrieved sample | **274** from **97** papers |
| Questions with mapped gold evidence (scored) | **253** (21 unmapped and excluded) |
| Gold paragraphs per scored question | mean **2.16**, min 1, **14 questions have more than 5** (so Recall@5 cannot reach 1.0 on them) |
| Retrieved documents per query | **top-5** |
| Best Recall@5 | **0.5365** (recursive + hybrid + cross-encoder rerank) |
| Paragraph-dense baseline (B0) Recall@5 | **0.5238** |
| Improvement over B0 | **+0.0127** absolute (+2.4% relative), 95% CI [−0.040, +0.065] |
| Best Hit@5 (at least one gold paragraph in top-5) | **0.727**, so **27%** of questions have no gold paragraph in the top-5 |
| Generated answers evaluated | **30** (11 papers) |
| RAGAS faithfulness completed | **27/30** |

## Setup

- **Dataset:** QASPER dev v0.3. Questions carry human-annotated supporting paragraphs and reference answers (extractive spans, free-form, yes/no, or unanswerable). The export contains 274 questions; 253 map to scored evidence at paragraph level.
- **Indexing:** one index **per paper** (so each question searches a single paper, not a corpus). Chunking strategies: paragraph, fixed, recursive, semantic, sentence window, parent-document, simplified auto-merging. Defaults: chunk 256, overlap 32.
- **Retrieval:** dense `BAAI/bge-small-en-v1.5`; BM25 (k1=1.5, b=0.75); hybrid by reciprocal-rank fusion (RRF). Default candidate pool 20 (`dense_top_k` = `sparse_top_k` = 10 unless stated).
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Query transformation:** none, rewrite, multi-query, HyDE. The run summaries record only `generator_model: qwen2.5:3b`; the transformation model is not recorded separately.
- **Generation:** `qwen2.5:3b` on the contexts from the best configuration, for the first 30 saved rows.
- **Judging:** local Ollama judge `qwen2.5:7b` for three local proxy scores, plus RAGAS metrics.
- **Retrieval metrics:** paragraph-level (`Recall@5`, `Precision@5`, `Hit@5`, `MRR`, `MAP`, `nDCG@5`). The metric credits **unique gold paragraphs**, so several chunks from one paragraph count once.

## How to read the numbers

- **B0 (baseline):** paragraph chunks + dense retrieval. Recall@5 0.5238, MRR 0.4443, nDCG@5 0.4096, Hit@5 0.696, about 7 ms per query.
- **Δ** is an absolute difference, not a percentage.
- **95% CI:** paired bootstrap over the 253 scored questions (5,000 resamples, fixed seed). **If an interval includes 0, the difference is not distinguishable from noise on this sample.** This is one run with one seed, so it does not capture run-to-run variation from LLM sampling or non-deterministic chunking.
- **Verdict labels:** **Reliable gain**, **Reliable loss**, **Not distinguishable from B0** (interval includes zero), **Not measurable here** (the method targets something retrieval metrics cannot see).

## Retrieval leaderboard: all 25 completed runs

Sorted by Recall@5, then MRR. Δ is versus B0. Runs with identical top-5 lists on all 253 questions are noted in the experiments below.

| # | Configuration | Key parameters | Recall@5 | ΔRecall@5 vs B0 [95% CI] | MRR | nDCG@5 | Hit@5 | Latency |
| - | ------------- | -------------- | -------- | ------------------------ | --- | ------ | ----- | ------- |
| 1 | Recursive / hybrid / rerank | chunk=256, RRF=60, cand=20, rerank_top_n=10 | 0.5365 | +0.013 [−0.040, +0.065] | 0.4850 | 0.4412 | 0.727 | 76.4 ms |
| 2 | Recursive / hybrid / rerank | chunk=256, RRF=60, cand=20, rerank_top_n=5 | 0.5365 | +0.013 [−0.040, +0.065] | 0.4850 | 0.4412 | 0.727 | 65.0 ms |
| 3 | Recursive / hybrid | chunk=256, RRF=30, dense/sparse_top_k=20/20 | 0.5283 | +0.004 [−0.042, +0.051] | 0.4615 | 0.4192 | 0.715 | 7.5 ms |
| 4 | Fixed / dense | chunk=512, overlap=32 | 0.5258 | +0.002 [0.000, +0.006] | 0.4483 | 0.4116 | 0.700 | 6.3 ms |
| 5 | Recursive / dense | chunk=256, dense_top_k=20 | 0.5248 | +0.001 [−0.015, +0.018] | 0.4445 | 0.4107 | 0.696 | 7.6 ms |
| 6 | Recursive / dense | chunk=256, dense_top_k=10 | 0.5248 | +0.001 [−0.015, +0.018] | 0.4445 | 0.4107 | 0.696 | 6.3 ms |
| 7 | Recursive / dense | chunk=512, dense_top_k=10 | 0.5238 | 0.000 [0.000, 0.000] | 0.4463 | 0.4101 | 0.696 | 6.9 ms |
| 8 | Paragraph / dense | candidate_k=15 | 0.5238 | baseline (B0) | 0.4443 | 0.4096 | 0.696 | 7.4 ms |
| 9 | Paragraph / dense | candidate_k=20 | 0.5238 | = B0 (identical top-5) | 0.4443 | 0.4096 | 0.696 | 6.8 ms |
| 10 | Recursive / hybrid | chunk=256, RRF=60, dense/sparse_top_k=10/10 | 0.5190 | −0.005 [−0.051, +0.042] | 0.4565 | 0.4135 | 0.712 | 7.7 ms |
| 11 | Fixed / dense | chunk=256, overlap=32 | 0.5171 | −0.007 [−0.021, +0.007] | 0.4413 | 0.4048 | 0.692 | 6.6 ms |
| 12 | Recursive / hybrid / multi-query | chunk=256, RRF=60, queries=3 | 0.5144 | −0.009 [−0.056, +0.038] | 0.4381 | 0.4068 | 0.688 | 1.26 s |
| 13 | Semantic / dense | percentile=90 | 0.5142 | −0.010 [−0.055, +0.035] | 0.4504 | 0.4140 | 0.688 | 7.2 ms |
| 14 | Semantic / dense | percentile=90 | 0.5141 | −0.010 [−0.045, +0.025] | 0.4533 | 0.4139 | 0.696 | 7.4 ms |
| 15 | Auto-merging / dense | merge=0.35, child=256 | 0.5132 | −0.011 [−0.058, +0.034] | 0.4336 | 0.4049 | 0.684 | 7.6 ms |
| 16 | Parent-document / dense | parent=512, child=256 | 0.5115 | −0.012 [−0.062, +0.035] | 0.4398 | 0.4093 | 0.680 | 7.6 ms |
| 17 | Parent-document / dense | parent=1024, child=256 | 0.5112 | −0.013 [−0.058, +0.031] | 0.4435 | 0.4106 | 0.680 | 7.3 ms |
| 18 | Auto-merging / dense | merge=0.60, child=256 | 0.5075 | −0.016 [−0.065, +0.029] | 0.4431 | 0.4068 | 0.680 | 7.4 ms |
| 19 | Recursive / hybrid / rewrite | chunk=256, RRF=60, rewrite | 0.5038 | −0.020 [−0.064, +0.026] | 0.4276 | 0.3924 | 0.676 | 288.9 ms |
| 20 | Recursive / hybrid / multi-query | chunk=256, RRF=60, queries=2 | 0.4985 | −0.025 [−0.076, +0.026] | 0.4484 | 0.3976 | 0.696 | 0.81 s |
| 21 | Recursive / hybrid / hyde | chunk=256, RRF=60, HyDE | 0.4881 | −0.036 [−0.089, +0.018] | 0.4159 | 0.3784 | 0.648 | 5.65 s |
| 22 | Sentence window / dense | window=3 | 0.4668 | −0.057 [−0.107, −0.009] | 0.3999 | 0.3690 | 0.628 | 8.8 ms |
| 23 | Sentence window / dense | window=1 | 0.4668 | −0.057 [−0.107, −0.009] | 0.3993 | 0.3685 | 0.628 | 8.9 ms |
| 24 | Recursive / BM25 | chunk=256, sparse_top_k=20 | 0.4306 | −0.093 [−0.151, −0.035] | 0.3673 | 0.3354 | 0.609 | 0.3 ms |
| 25 | Recursive / BM25 | chunk=256, sparse_top_k=10 | 0.4299 | −0.094 [−0.152, −0.036] | 0.3692 | 0.3357 | 0.609 | 0.2 ms |

**Exact run IDs, in leaderboard order**

1. `recursive_hybrid_none_rerank_bd0feccaf82a`
2. `recursive_hybrid_none_rerank_941727ac31a6`
3. `recursive_hybrid_none_raw_579ce42d57d7`
4. `fixed_dense_none_raw_33d59b9de9aa`
5. `recursive_dense_none_raw_04818043eb3f`
6. `recursive_dense_none_raw_07bf33c4b819`
7. `recursive_dense_none_raw_97fdd06a2ea1`
8. `paragraph_dense_none_raw_0283a68906b3`
9. `paragraph_dense_none_raw_ccd47dc3f09c`
10. `recursive_hybrid_none_raw_3f1778529753`
11. `fixed_dense_none_raw_bd82b0a1f44f`
12. `recursive_hybrid_multi_query_raw_3a057d3da38c`
13. `semantic_dense_none_raw_1bfe3ed3280e`
14. `semantic_dense_none_raw_c140ffeb308b`
15. `auto_merging_dense_none_raw_fecba2351f90`
16. `parent_dense_none_raw_e3ee3bc5f48b`
17. `parent_dense_none_raw_9e7c48d53d78`
18. `auto_merging_dense_none_raw_e9de941cb339`
19. `recursive_hybrid_rewrite_raw_eedacefe9e5e`
20. `recursive_hybrid_multi_query_raw_3804b7abb907`
21. `recursive_hybrid_hyde_raw_70b1e8d137e3`
22. `sentence_window_dense_none_raw_18ee611b8977`
23. `sentence_window_dense_none_raw_b5e3fa5506ac`
24. `recursive_bm25_none_raw_51a2f48b860e`
25. `recursive_bm25_none_raw_c1afb95b5778`

---

## Experiments: what we tried, what it did, and why

Evidence means something the saved runs show. **Hypothesis** means an explanation we have not tested; each comes with the check that would test it.

### Summary of all experiments

| # | Experiment | What changed | Result | Verdict |
| - | ---------- | ------------ | ------ | ------- |
| E1a | Recursive / fixed chunking | Splitter and size (256 or 512) | Within ±0.007 Recall@5 of B0 | Not distinguishable from B0 |
| E1b | Semantic chunking | Split on embedding-similarity percentile | About −0.010 Recall@5, MRR slightly up | Not distinguishable from B0 |
| E1c | Sentence window | Index single sentences, expand to a window | About −0.057 Recall@5 | **Reliable loss** |
| E1d | Parent-document / auto-merging | Retrieve small chunks, return larger parent | About −0.011 to −0.016 Recall@5 | Not measurable here (and not distinguishable) |
| E2 | Hybrid BM25 + dense (RRF) | Fuse sparse and dense rankings | −0.005 (RRF=60) or +0.005 (RRF=30) Recall@5 | Not distinguishable from B0 |
| E3 | BM25 only | Sparse retrieval alone | About −0.093 Recall@5 | **Reliable loss** |
| E4 | Cross-encoder reranking | Rerank hybrid candidates | +0.013 Recall@5, +0.041 MRR | Not distinguishable from B0 (best nominal score) |
| E5 | Query transformation | Rewrite / multi-query / HyDE with an LLM | −0.005 to −0.031 Recall@5 vs raw hybrid, 37× to 734× slower | Not distinguishable, but never positive |

---

### E1. Chunking strategy (dense retrieval, nothing else changed)

**Question:** does how we cut a paper into retrievable units matter?

| Variant | Recall@5 | ΔRecall@5 vs B0 [95% CI] | MRR |
| ------- | -------- | ------------------------ | --- |
| Paragraph (B0) | 0.5238 | | 0.4443 |
| Recursive 256 | 0.5248 | +0.001 [−0.015, +0.018] | 0.4445 |
| Recursive 512 | 0.5238 | 0.000 [0.000, 0.000] | 0.4463 |
| Fixed 512 / overlap 32 | 0.5258 | +0.002 [0.000, +0.006] | 0.4483 |
| Fixed 256 / overlap 32 | 0.5171 | −0.007 [−0.021, +0.007] | 0.4413 |
| Semantic p90 (run 1) | 0.5142 | −0.010 [−0.055, +0.035] | 0.4504 |
| Semantic p90 (run 2) | 0.5141 | −0.010 [−0.045, +0.025] | 0.4533 |
| Parent-document 512 | 0.5115 | −0.012 [−0.062, +0.035] | 0.4398 |
| Parent-document 1024 | 0.5112 | −0.013 [−0.058, +0.031] | 0.4435 |
| Auto-merging 0.35 | 0.5132 | −0.011 [−0.058, +0.034] | 0.4336 |
| Auto-merging 0.60 | 0.5075 | −0.016 [−0.065, +0.029] | 0.4431 |
| Sentence window 1 | 0.4668 | −0.057 [−0.107, −0.009] | 0.3993 |
| Sentence window 3 | 0.4668 | −0.057 [−0.107, −0.009] | 0.3999 |

**Result:** paragraph, recursive and fixed chunking are indistinguishable. The structured variants (semantic, parent-document, auto-merging) are slightly lower on recall, inside the noise. **Sentence window is the only chunking choice that is reliably worse.**

**Why (Evidence):**
- Sentence window 1 and 3 return the same top-5 on 250 of 253 questions, so the window size does not change what retrieval returns. The window expansion happens after retrieval and is invisible to these metrics.
- Sentence-window top-5 lists contain on average **4.47 distinct paragraphs** (B0: 5.00), because several of the top-5 sentences come from the same paragraph and the metric credits each gold paragraph once. This explains part of the loss, not all of it.
- Dense retrieval depth does not matter here: recursive-256 runs with `dense_top_k` of 10 and 20 return identical top-5 lists on all 253 questions, and so do paragraph runs with candidate pools of 15 and 20.
- Two identically configured semantic-chunking runs return different top-5 lists on 239 of 253 questions yet score within 0.0001 Recall@5, so semantic chunking is non-deterministic and the true run-to-run noise on individual questions is large.

**Why (Hypothesis):** single-sentence embeddings are too short to match a question well. **Check:** compare sentence-window retrieval after de-duplicating hits to the same paragraph.

**Takeaway:** chunking is not the lever on this benchmark. Use the cheapest strategy (paragraph or recursive 256). Parent-document and auto-merging need a matched **generation** evaluation before they can be judged, since they exist to give the generator more context.

---

### E2. Hybrid retrieval: BM25 + dense with reciprocal-rank fusion

| Variant | Recall@5 | ΔRecall@5 vs B0 [95% CI] | MRR | ΔMRR vs B0 [95% CI] | Latency |
| ------- | -------- | ------------------------ | --- | ------------------- | ------- |
| Dense (B0) | 0.5238 | | 0.4443 | | 7 ms |
| Hybrid RRF=60, pool 10+10 | 0.5190 | −0.005 [−0.051, +0.042] | 0.4565 | +0.012 [−0.026, +0.050] | 7.7 ms |
| Hybrid RRF=30, pool 20+20 | 0.5283 | +0.004 [−0.042, +0.051] | 0.4615 | +0.017 [−0.020, +0.056] | 7.5 ms |

**Result:** hybrid retrieval does not move Recall@5 (it is slightly lower at RRF=60 and slightly higher at RRF=30, both inside the noise) and shifts MRR up by about +0.012 to +0.017, also inside the noise.

**Why (Evidence):** the RRF=30 run also doubled the candidate pool (`dense_top_k` and `sparse_top_k` 10 → 20), so **RRF=30 vs RRF=60 is not a clean ablation.** The two runs return different top-5 lists on 121 of 253 questions.

**Why (Hypothesis):** dense and BM25 make different mistakes, so fusion helps push agreed paragraphs up the ranking, but it cannot add paragraphs that neither retriever surfaces. **Check:** repeat with the pool held fixed at 10+10 and 20+20 for both RRF constants.

---

### E3. BM25 only

| Variant | Recall@5 | ΔRecall@5 vs B0 [95% CI] | MRR | ΔMRR vs B0 [95% CI] | Latency |
| ------- | -------- | ------------------------ | --- | ------------------- | ------- |
| BM25, sparse_top_k=20 | 0.4306 | −0.093 [−0.151, −0.035] | 0.3673 | −0.077 [−0.128, −0.025] | 0.28 ms |
| BM25, sparse_top_k=10 | 0.4299 | −0.094 [−0.152, −0.036] | 0.3692 | | 0.23 ms |

**Result:** about 9 points of Recall@5 lost against dense, roughly 25× faster. **A reliable loss.** The "0.000 s" shown in the dashboard is three-decimal rounding, not zero time.

**Why (Hypothesis):** QASPER questions are written by readers who often paraphrase the paper, so exact keyword matching misses paragraphs that dense embeddings find. **Check:** bucket questions by word overlap with their gold paragraph and compare BM25 and dense per bucket. E2 shows BM25 still contributes as a fusion partner, but the effect is within noise.

---

### E4. Cross-encoder reranking (best nominal configuration)

| Variant | Recall@5 | ΔRecall@5 vs B0 [95% CI] | MRR | nDCG@5 | Latency |
| ------- | -------- | ------------------------ | --- | ------ | ------- |
| Hybrid RRF=60, no rerank | 0.5190 | −0.005 [−0.051, +0.042] | 0.4565 | 0.4135 | 7.7 ms |
| + rerank, `rerank_top_n=5` | 0.5365 | +0.013 [−0.040, +0.065] | 0.4850 | 0.4412 | 65.0 ms |
| + rerank, `rerank_top_n=10` | 0.5365 | +0.013 [−0.040, +0.065] | 0.4850 | 0.4412 | 76.4 ms |

- **Versus B0:** ΔRecall@5 +0.013 [−0.040, +0.065], ΔMRR +0.041 [−0.007, +0.089].
- **Versus the same hybrid without reranking:** ΔRecall@5 +0.018 [−0.021, +0.055], ΔMRR +0.029 [−0.013, +0.070].
- **Per-question:** against B0 the reranked run is better on 49 questions, tied on 162 and worse on 42.

**Result:** the highest scores in the sweep, but **every interval includes zero**. The effect is larger on MRR (+0.041, +9% relative) than on Recall@5 (+0.013, +2.4% relative), which fits a reranker that mainly improves where the right paragraph sits in the list.

**Why (Evidence):**
- `rerank_top_n=5` and `10` return **identical top-5 lists on all 253 questions**: they are the same experiment run twice. The latency difference (65.0 vs 76.4 ms) is timing noise or caching, not an effect of the parameter.
- The reranker costs about 10× the latency of un-reranked hybrid (7.7 ms to 76.4 ms), and the run took 671 s versus about 30 s for raw hybrid.

**Why (Hypothesis):** a cross-encoder reads the question and paragraph jointly, so it separates "mentions the topic" from "answers the question" better than a bi-encoder. **Check:** run the full 1,005-question QASPER dev set; with roughly 3.7× the questions, intervals shrink by about half, which would settle whether a +0.01 to +0.04 effect is real.

**Takeaway:** worth trying if the top 1 to 2 passages are what reach the generator. On this sample the evidence does not justify 10× latency for top-5 recall.

---

### E5. Query transformation (rewrite, multi-query, HyDE)

All variants use recursive hybrid with RRF=60 and no rerank, compared with the raw-query run (Recall@5 0.5190, MRR 0.4565, 7.7 ms/query, 30 s total).

| Variant | Recall@5 | ΔRecall@5 vs raw [95% CI] | MRR | ΔMRR vs raw [95% CI] | Latency | Total run time |
| ------- | -------- | ------------------------- | --- | -------------------- | ------- | -------------- |
| Raw query | 0.5190 | | 0.4565 | | 0.0077 s | 30 s |
| Rewrite | 0.5038 | −0.015 [−0.054, +0.025] | 0.4276 | −0.029 [−0.062, +0.005] | 0.289 s (37×) | 101 s |
| Multi-query, 2 | 0.4985 | −0.020 [−0.062, +0.019] | 0.4484 | −0.008 [−0.042, +0.026] | 0.815 s (106×) | 242 s |
| Multi-query, 3 | 0.5144 | −0.005 [−0.042, +0.034] | 0.4381 | −0.018 [−0.055, +0.018] | 1.264 s (164×) | 361 s |
| HyDE | 0.4881 | −0.031 [−0.078, +0.016] | 0.4159 | −0.041 [−0.093, +0.010] | 5.649 s (734×) | 1479 s |

**Result:** no transformation beat the raw query on Recall@5, and none beat it on MRR. The intervals include zero, so this is "never positive" rather than "reliably negative". The cost is not in doubt: HyDE took about 25 minutes versus 30 seconds.

**Why (Evidence):** multi-query with 3 reformulations (0.5144) scores higher than with 2 (0.4985), which is the wrong direction for a real trend and points to variance from LLM sampling.

**Why (Hypothesis):**
- Indexes are **per paper**, so each question searches a small pool; query expansion mainly helps when the corpus is large and the query is vague.
- A small local LLM can drift from the question (rewrite) or embed a fabricated answer (HyDE). **Check:** repeat at temperature 0 with three seeds and a larger transformation model.

**Takeaway:** skip query transformation on this benchmark.

---

## Retrieval by question type, and the recall ceiling

Best configuration versus B0, split by the answer type annotators gave (majority over annotators; ties go to the first annotator):

| Question type (annotator majority) | n | Recall@5 B0 | Recall@5 best | ΔRecall@5 [95% CI] | MRR B0 | MRR best | ΔMRR [95% CI] |
| ---- | - | ----------- | ------------- | ----------------- | ------ | -------- | ------------ |
| Extractive (spans) | 145 | 0.572 | 0.561 | −0.011 [−0.078, +0.054] | 0.500 | 0.536 | +0.036 [−0.026, +0.096] |
| Free-form (abstractive) | 61 | 0.502 | 0.518 | +0.016 [−0.094, +0.127] | 0.434 | 0.455 | +0.021 [−0.097, +0.136] |
| Yes / no | 33 | 0.362 | 0.479 | +0.117 [−0.017, +0.250] | 0.282 | 0.370 | +0.088 [−0.019, +0.196] |
| Unanswerable | 14 | 0.500 | 0.500 | 0.000 [−0.321, +0.321] | 0.290 | 0.357 | +0.067 [−0.238, +0.350] |

- **Yes/no questions show the largest apparent gain** (Recall@5 0.362 to 0.479), but there are only 33 of them and the interval includes zero. Treat as a lead for follow-up, not a finding.
- **Multi-evidence questions are hard for any top-5 retriever.** For the best configuration, questions with 1 gold paragraph average 0.683 Recall@5 (123 questions), with 2 gold paragraphs 0.473 (73), with 3 gold paragraphs 0.273 (22), and with 4 or more 0.321 (35).
- **Longer questions retrieve slightly worse** (Recall@5 0.57 for 1 to 5 words, 0.41 for 13 or more; rank correlation −0.11, p = 0.09), which is weak evidence only.

---

## Answer quality: 30 generated answers

The 30 rows come from **one** retrieval configuration (the best one), not one set per method, and cover 11 papers. They are the first 30 saved rows, so they are not a random sample.

| Metric | Valid / 30 | Mean | What it is |
| ------ | ---------- | ---- | ---------- |
| Groundedness | 30/30 | **0.523** | Local judge: support from context |
| Context relevance | 30/30 | **0.690** | Local judge: question vs contexts |
| Answer relevance | 30/30 | **0.700** | Local judge: answer vs question |
| Faithfulness | 27/30 | **0.783** | RAGAS: answer claims supported by context |
| Context precision | 30/30 | **0.553** | RAGAS: ranked contexts vs reference |
| Context recall | 30/30 | **0.560** | RAGAS: reference coverage by contexts |
| Answer relevancy | 30/30 | **0.646** | RAGAS: answer vs question |
| Reference token F1 | 30/30 | **0.122** | Lexical overlap with the reference |

Questions are numbered **Q1 to Q30** (the 1-based row in `generation_scores.csv`).

### Manual correctness review

The automatic scores do not measure correctness against the reference, so we read all 30 answers against the QASPER reference answers and labeled each one. These are **human labels from a single reviewer**, so some borderline calls (especially "partial") could go the other way; the full table is in the appendix.

| Label | Count | Mean Recall@5 | Gold paragraph in top-5 | Mean RAGAS faithfulness | Mean token F1 |
| ----- | ----- | ------------- | ----------------------- | ----------------------- | ------------- |
| Correct | 11 | 0.855 | 11 of 11 | 0.871 | 0.138 |
| Partially correct (generic, or missing part) | 7 | 0.571 | 7 of 7 | 0.827 | 0.170 |
| Wrong (wrong or invented content, or off-target) | 8 | 0.119 | 3 of 8 | 0.744 | 0.101 |
| Abstained ("cannot determine") when the reference had an answer | 4 | 0.062 | 1 of 4 | 0.562 | 0.035 |

What this shows:

- **No answer was correct without a gold paragraph in the top-5.** Of the 12 wrong or abstained answers, **8 had zero gold evidence retrieved** (retrieval failure) and **4 had some** (generation failure: Q8, Q20, Q23, Q24).
- **Token F1 does not separate correct from partial answers** (0.138 vs 0.170), and RAGAS faithfulness barely separates correct from wrong (0.871 vs 0.744). Neither is a correctness metric; a hedged or abstaining answer is often "faithful".
- The 3B generator **abstains or hedges** on questions whose answer is "no" or "none" (Q7, Q10), where QASPER references state a negative fact.

### Judge reliability

- **RAGAS faithfulness is missing for Q11, Q15 and Q20**, where the judge output was cut off by `max_tokens` (`IncompleteOutputException` after two attempts). These are missing, not zero.
- **The local judge returned exactly 0/0/0 on six rows (Q2, Q5, Q10, Q12, Q15, Q27).** For Q10, Q12 and Q27 that matches a real failure, but for Q2, Q5 and Q15 the answers are correct or partially correct and RAGAS context precision is 0.89 to 1.0, which looks like a judge output failure. Excluding all six rows would raise the mean groundedness from 0.523 to 0.654, context relevance from 0.690 to 0.863 and answer relevance from 0.700 to 0.875; we report the unadjusted values above because the six rows cannot be cleanly separated.
- The local judge also gave a groundedness of 0.8 to answers that RAGAS scored 0.33 faithful (Q9) or whose content is invented (Q20), so it is a weak check for unsupported claims.

### Example gallery

Each example shows the saved question, reference, an excerpt of the generated answer, and the scores. "Local" is the local judge's groundedness; "F1" is reference token F1.

#### A. Correct answers

**Q21 · free-form · correct, well grounded.** *How do they calculate a static embedding for each word?*
- **Reference:** the first principal component of a word's contextualized representation in a given layer.
- **Answer:** "…static embeddings are created by taking the first principal component (PC) of the contextualized representations of a word in a given layer."
- **Scores:** Recall@5 1.00 · MRR 1.00 · local 0.8 · faithfulness 1.00 · context recall 1.00 · F1 0.12.
- **Why it matters:** retrieval found the evidence and the generator restated it. F1 is only 0.12 because the answer is long.

**Q30 · extractive · correct, quotes its source.** *Who were the human evaluators used?*
- **Reference:** 20 evaluators recruited from the authors' institution, each doing 20 annotations.
- **Answer:** quotes the evidence directly and concludes the evaluators were sourced from the authors' institution.
- **Scores:** Recall@5 1.00 · MRR 1.00 · local 0.5 · faithfulness 1.00 · F1 0.44.
- **Why it matters:** a verbatim, correct answer still got a local groundedness of only 0.5.

**Q16 · extractive list · correct, but F1 says otherwise.** *On which benchmarks they achieve the state of the art?*
- **Reference:** SimpleQuestions, WebQSP.
- **Answer:** names "a new state-of-the-art result on SimpleQuestions and a result comparable to the state-of-the-art on WebQSP".
- **Scores:** Recall@5 1.00 · MRR 0.25 · faithfulness 1.00 · F1 **0.03**.
- **Why it matters:** both benchmarks are named, yet token F1 is near zero, because the answer buries them in a long explanation.

#### B. Partially correct

**Q15 · extractive · too generic.** *Which information about typography is included in the corpus?*
- **Reference:** font type and font style (italics, bold), and the token's position on the page.
- **Answer:** "The corpus includes information on text structure, typography, and images." It repeats the question's own word instead of naming font type and style.
- **Scores:** Recall@5 0.25 · MRR 0.20 · RAGAS context precision 0.89, context recall 1.00 · local judge 0/0/0 · faithfulness missing (truncated judge output) · F1 0.15.
- **Why it matters:** the evidence was in the retrieved contexts (context recall 1.00), but the generator did not extract the specifics. Q14 (text structure) shows the same pattern.

**Q4 · list · one invented item.** *What language pairs are explored?*
- **Reference:** Fr-En-Es, De-En-Fr, Ro-En-De, plus Arabic, Spanish and Russian with mutual translation.
- **Answer:** lists three correct triples and adds "Arabic-English-Spanish (Ar-En-Es)", which is not what the reference describes.
- **Scores:** Recall@5 0.50 · MRR 0.20 · faithfulness 0.80 · context recall 0.33 · F1 0.36.
- **Why it matters:** partial retrieval (Recall@5 0.50) led to a partly right, partly extrapolated list.

#### C. Wrong because the gold paragraph was not retrieved

**Q12 · numeric · missing the requested number.** *On how many language pairs do they show that preordering assisting language sentences helps translation quality?*
- **Reference:** 5.
- **Answer:** a general statement that preordering "significantly improves translation quality", with no count.
- **Scores:** Recall@5 0.00 · local 0.0 · faithfulness 0.50 · F1 0.06.

**Q13 · extractive · answers a different question.** *Which dataset(s) do they experiment with?*
- **Reference:** the IITB English-Hindi parallel corpus and the ILCI English-Hindi parallel corpus.
- **Answer:** describes the translation tasks and source languages (Bengali, Gujarati, Marathi, Malayalam, Tamil) instead of naming datasets.
- **Scores:** Recall@5 0.00 · local 0.2 · faithfulness 0.75 · context recall 0.00 · F1 0.09.

**Q9 · extractive · generic instead of the model name.** *What predictive model do they build?*
- **Reference:** logistic regression models (bag-of-words, sentiment, linguistic features).
- **Answer:** says a model "is trained to classify dogmatic posts from Reddit" and never names the model type.
- **Scores:** Recall@5 0.00 · local **0.8** · faithfulness **0.33** · context recall 0.00 · F1 0.10.
- **Why it matters:** the local judge and RAGAS disagree sharply on the same answer.

#### D. Wrong although some gold evidence was retrieved (generation failures)

**Q24 · yes/no · answer inverted.** *Does BERT reach the best performance among all the algorithms compared?*
- **Reference:** no.
- **Answer:** "BERT does appear to reach the best performance among the compared algorithms," citing an F1 margin.
- **Scores:** Recall@5 0.25 · local 0.8 · **faithfulness 1.00** · context recall 0.00 · F1 0.00.
- **Why it matters:** faithful to the retrieved passages, wrong against the paper. Faithfulness cannot detect this.

**Q8 · extractive list · picked the wrong topics.** *What are the topics pulled from Reddit?*
- **Reference:** politics, business, science, AskReddit, plus posts from the Reddit frontpage.
- **Answer:** "…including but not limited to climate change, abortion, world news, relationship advice, and the future of artificial intelligence."
- **Scores:** Recall@5 0.50 · local 0.2 · **faithfulness 1.00** · context recall 1.00 · F1 0.13.
- **Why it matters:** RAGAS says the contexts cover the reference (context recall 1.00), yet the 3B generator listed other topics mentioned nearby.

#### E. Not grounded (speculation or invented content)

**Q6 · extractive · speculates, then concludes confidently.** *What is the source of the news sentences?*
- **Reference:** ilur.am; links between Wikipedia articles used to generate annotated sequences.
- **Answer:** concedes the evidence "does not explicitly state" Wikipedia as the source, then argues the sentences are "likely sourced from Wikipedia and manually annotated news texts."
- **Scores:** Recall@5 0.50 · MRR 1.00 · local 0.7 · **faithfulness 0.25** · F1 0.08.
- **Why it matters:** the lowest faithfulness in the set for an answer that is partly right, a pattern of hedged reasoning stated as a conclusion.

**Q20 · free-form · invents its own experiments.** *What experiments are proposed to test that upper layers produce context-specific embeddings?*
- **Reference:** the paper measures self-similarity, intra-sentence similarity and maximum explainable variance in the upper layers.
- **Answer:** "several experiments can be proposed… Experiment 1: Layer-wise Analysis of Context-Specificity…", a designed protocol rather than what the paper did.
- **Scores:** Recall@5 0.20 · MRR 0.20 · **local 0.8** · faithfulness **missing** (judge output truncated) · F1 0.16.
- **Why it matters:** the local judge rated invented content 0.8 and RAGAS did not score it, so no metric flagged it. Only reading the answer did.

#### F. Abstained when the reference had an answer

**Q7 · yes/no · "cannot determine" instead of "no".** *Did they use a crowdsourcing platform for manual annotations?*
- **Reference:** no.
- **Answer:** "…we cannot determine if they used a crowdsourcing platform… The evidence does not mention any use of crowdsourcing."
- **Scores:** Recall@5 0.00 · local 0.2 · faithfulness 0.67 · F1 0.00.
- **Why it matters:** the answer is nearly right in substance (no evidence of crowdsourcing) but fails as a yes/no. Q10 is the open-ended version: reference says no platform was used (students were hired), and the answer says the platform cannot be determined.

**Q27 · extractive · a generic question with no retrieval hit.** *What was the baseline?*
- **Reference:** the system presented by deri2016grapheme; wFST.
- **Answer:** "…we cannot definitively determine what the baseline was…"
- **Scores:** Recall@5 0.00 · all local scores 0 · faithfulness 0.50 · F1 0.00.

#### G. Judge disagreement (the answer is fine, the score is not)

**Q5 · extractive · correct answer, zero local scores.** *What NER models were evaluated?*
- **Answer:** Stanford NER, spaCy 2.0 and a recurrent model, matching the reference.
- **Scores:** local groundedness, context relevance and answer relevance all **0.0**; RAGAS faithfulness 1.00, context precision 1.00, context recall 1.00; Recall@5 0.50, MRR 1.00.

**Q2 · extractive · same pattern.** *What are the pivot-based baselines?* The answer names "pivoting" (correct) but adds "pivot-synthetic" where the reference lists the pivoting variants. The local judge gives 0/0/0, while RAGAS gives faithfulness 1.00 and context recall 1.00.

---

## Question types and what is missing

| Type | In retrieval sample (253) | Among the 30 answers | Coverage |
| ---- | ------------------------- | -------------------- | -------- |
| Extractive | 145 | about 20 | Well covered |
| Free-form (abstractive) | 61 | about 6 | Thin |
| Yes / no | 33 | **2** (Q7, Q24), both failed | Anecdotal |
| Unanswerable | 14 | **0 clear cases** (Q10 and Q11 were marked unanswerable by one annotator and answered by another) | **Not tested** |
| Multi-evidence (more than one gold paragraph) | 130 | not tabulated | Retrieval only |

Counts in the middle column use the annotator majority; the right column uses the first annotator, so they do not add up exactly. The generation set has no clean **unanswerable** question, so we do not know whether the generator correctly refuses when the paper has no answer. Given that it already abstains when a "no" is correct (Q7, Q10), this is the most important gap to close.

---

## Outliers, failures and limitations

1. **Small, noisy sample.** 253 scored questions, one seed: only BM25-only and sentence-window differences clear the noise floor. See the confidence intervals above.
2. **Experiments are not a fully controlled ablation.** RRF=30 also changed the candidate pool (E2). Several pairs of runs are duplicates in practice: `rerank_top_n=5` vs `10`, `dense_top_k=10` vs `20`, and paragraph candidate pools of 15 vs 20 return identical top-5 lists, so they add no information.
3. **Evidence mapping:** 21 of 274 sampled questions have no scored evidence and are excluded. The run summaries also report an `unanswerable: 40` count whose definition is not documented in the artifacts; by annotator majority only 14 of the 253 scored questions are unanswerable.
4. **Generation covers one configuration and 30 questions.** No answer-quality comparison between retrieval methods can be made.
5. **Correctness is not measured by the automatic metrics.** The manual labels in this README are one reviewer's judgment; reference-aware correctness, exact-match for short answers and yes/no accuracy are missing.
6. **Incomplete judge runs:** RAGAS faithfulness is missing for Q11, Q15 and Q20 (`max_tokens` truncation). The local judge's 0/0/0 outputs look like parse failures on at least three rows.
7. **Timings** are per-question means from this run, not a cross-hardware benchmark, and exclude model loading and index building.
8. **QASPER references** often have two acceptable answers; the RAGAS export uses a single reference string for some metrics.

## Recommended next experiments

1. **Run the full QASPER dev set (1,005 questions)** with three seeds, and keep the bootstrap intervals. This is the single change most likely to turn "not distinguishable" into a verdict.
2. **Hold the candidate pool fixed** (10+10 and 20+20) while varying the RRF constant, and drop duplicate runs from the sweep.
3. **Matched generation on 100+ questions** for rerank-hybrid, raw hybrid, paragraph-dense and auto-merging. Score correctness, not only faithfulness.
4. **Fix the judging pipeline:** raise the RAGAS `max_tokens`, retry only the 3 missing faithfulness rows, and add a reference-aware correctness metric (normalized exact match for short answers, yes/no accuracy, semantic equivalence).
5. **Test abstention:** include unanswerable questions in the generation set, and compare the 3B generator with a larger one on yes/no and "none" answers.
6. **Query-type breakdown:** the yes/no gain from reranking (+0.117 Recall@5, interval includes zero) is worth confirming on the full set.

## Running locally

```
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
python analyze_runs.py artifacts      # regenerates the retrieval statistics in this README
```

Run the RAGxiv notebook to build indexes, execute the sweep, generate answers, and run the optional RAGAS judge. Exported artifacts:

```
artifacts/
├── runs/
│   ├── <run_id>_summary.json   # configuration, retrieval metrics, timings
│   └── <run_id>.json           # question-level evidence and contexts
├── analysis/
│   ├── leaderboard_ci.csv      # paired bootstrap intervals (from analyze_runs.py)
│   └── by_type.csv             # retrieval by question type
├── generation_scores.csv       # questions, references, answers, local judge scores
├── ragas_scores.csv            # RAGAS metrics
├── qasper-dev-v0.3.json        # dataset (about 11 MB)
└── sweep_failures.json         # []
```

> **Publishing note:** `artifacts/` holds paper passages, generated answers and a copy of the QASPER dev file. Review dataset licensing, file sizes (the archive is about 47 MB uncompressed, including a second dataset tarball) and attribution before committing raw `artifacts/` to a public repository. Consider ignoring the dataset files and linking to QASPER instead.

## Bottom line

**On this sample, a plain paragraph-dense retriever is hard to beat.** Recursive chunking, hybrid retrieval and cross-encoder reranking give the best nominal scores (Recall@5 0.5365 vs 0.5238), but the difference is within noise. The two reliable findings are what *not* to do: BM25-only and sentence-window indexing lose recall, and LLM query rewriting, multi-query and HyDE add cost without a gain. For answers, retrieval is a hard ceiling (no correct answer without a gold paragraph in the top-5), but it is not enough: a faithful answer can still contradict the paper, as in Q24, and no automatic metric here measures that. The most valuable next steps are more questions, matched generation runs, and a correctness metric.

---

## Appendix: all 30 answers, labeled

Labels are our manual review against the QASPER references. R@5 is retrieval Recall@5 for that question; "Faith." is RAGAS faithfulness (— = missing).

| Q | Question (short) | Label | R@5 | Faith. | Note |
| - | ---------------- | ----- | --- | ------ | ---- |
| 1 | Multilingual approaches compared | Correct | 1.00 | 1.00 | Names MNMT; adds pivoting and transfer baselines |
| 2 | Pivot-based baselines | Partial | 0.75 | 1.00 | "Pivoting" right; second baseline differs |
| 3 | Datasets experimented with | Correct | 1.00 | 0.50 | Europarl, MultiUN |
| 4 | Language pairs explored | Partial | 0.50 | 0.80 | One invented triple (Ar-En-Es) |
| 5 | NER models evaluated | Correct | 0.50 | 1.00 | Stanford NER, spaCy, recurrent model |
| 6 | Source of news sentences | Partial | 0.50 | 0.25 | Hedged, speculates Wikipedia |
| 7 | Crowdsourcing platform used? (yes/no) | Abstained | 0.00 | 0.67 | "Cannot determine"; reference is no |
| 8 | Topics pulled from Reddit | Wrong | 0.50 | 1.00 | Lists other topics from the context |
| 9 | Predictive model built | Wrong | 0.00 | 0.33 | Never names logistic regression |
| 10 | Crowdsourcing platform used | Abstained | 0.00 | 0.75 | Reference: no platform, hired students |
| 11 | How words are matched before reordering | Correct | 1.00 | — | CFILT-preorder system |
| 12 | Number of language pairs | Wrong | 0.00 | 0.50 | No number given (reference: 5) |
| 13 | Datasets used | Wrong | 0.00 | 0.75 | Describes language pairs, not datasets |
| 14 | Text structure information | Partial | 0.50 | 1.00 | Generic; misses paragraph/line segmentation |
| 15 | Typography information | Partial | 0.25 | — | Generic; misses font type and style |
| 16 | Benchmarks with state of the art | Correct | 1.00 | 1.00 | SimpleQuestions and WebQSP (hedged) |
| 17 | What the proposed framework uses | Wrong | 0.00 | 0.63 | Wrong components (relation detectors) |
| 18 | What KBQA abbreviates | Correct | 1.00 | 1.00 | Knowledge Base Question Answering |
| 19 | Core component for KBQA | Partial | 0.50 | 0.91 | "Relation detection model"; fits loosely |
| 20 | Experiments proposed for upper layers | Wrong | 0.20 | — | Invents its own experiments |
| 21 | Static embedding calculation | Correct | 1.00 | 1.00 | First principal component |
| 22 | BERT performance | Wrong | 0.00 | 1.00 | "Outstanding", no F1 numbers |
| 23 | Other algorithms tested | Abstained | 0.25 | 0.33 | "Not specified"; reference: CRF, spaCy |
| 24 | BERT best among all? (yes/no) | Wrong | 0.25 | 1.00 | Says yes; reference is no |
| 25 | Clinical datasets used | Correct | 0.50 | 0.50 | MEDDOCAN and NUBes |
| 26 | How model compactness is measured | Correct | 1.00 | 1.00 | File size on disk |
| 27 | What was the baseline | Abstained | 0.00 | 0.50 | "Cannot determine"; reference: wFST |
| 28 | Evaluation metrics | Correct | 0.40 | 0.71 | WER, WER-100, PER |
| 29 | Datasets used (g2p) | Partial | 1.00 | 1.00 | One of three sources; accepted by one reference |
| 30 | Human evaluators used | Correct | 1.00 | 1.00 | 20 evaluators from the authors' institution |
