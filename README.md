# RAGxiv — Retrieval & Answer Quality Experiments

> **Reproducible results snapshot.** This report is derived from the exported `artifacts/` directory, not manually transcribed dashboard plots. The results describe this run of the QASPER benchmark; they are not general rankings of RAG methods.

## At a glance

| Measure | Observed result |
|---|---:|
| Completed retrieval configurations | **25** |
| QASPER questions in the retrieved evaluation set | **274** |
| Questions with mapped gold evidence (per run) | **253** |
| Excluded from scored retrieval due to unmapped evidence | **21** |
| Retrieved documents returned per query | **top-5** |
| Best retrieval Recall@5 | **0.5365** |
| Best method | **Recursive chunks + hybrid BM25/dense + cross-encoder reranking** |
| Paragraph-dense baseline Recall@5 | **0.5238** |
| Improvement over paragraph-dense | **+0.0127 absolute** (+2.4% relative) |
| Generated answers evaluated | **30** |
| RAGAS faithfulness completed | **27/30** |

## Experiment setup

- **Dataset:** QASPER research-paper questions with human-annotated supporting evidence and reference answers. The export contains 274 questions in the retrieval sample, of which 253 have evaluable gold-evidence mappings. The configured paper cap may exceed the number of papers actually represented in the sample.
- **Indexing / retrieval:** paper-level indexes over varied chunking strategies (paragraph, fixed, recursive, semantic, sentence window, parent-document, simplified auto-merging); dense embeddings `BAAI/bge-small-en-v1.5`, BM25 sparse retrieval, and reciprocal-rank-fusion hybrid retrieval.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranked configurations.
- **Query transformations:** none, rewriting, multiple reformulations, and hypothetical-document embeddings (HyDE); some transformations invoke a local LLM.
- **Generation:** `qwen2.5:3b` using the passages retrieved by the leading reranked configuration, for the first 30 saved evaluation rows.
- **Judging:** local Ollama judge `qwen2.5:7b` for local proxy scores and the separately exported RAGAS metrics.
- **Metric meaning:** retrieval metrics use paragraph-level mapped evidence (`Recall@5`, `Precision@5`, `MRR`, `MAP`, `nDCG`). Generation metrics use model judgment and reference-overlap; they should not be confused with direct factual correctness.
- **Comparability:** all 25 retrieval summaries report 253 scored questions and 21 unmapped questions. Scores are averaged over the scored questions; query latency is the reported mean per evaluated question from the run summary.

## Retrieval leaderboard — all completed runs

**Sorted by mean Recall@5.** Different entries can vary parameters beyond the method label; the run ID is the stable identifier for the exact saved experiment.

| # | Configuration | Important parameter | Recall@5 | MRR | nDCG@5 | Query latency |
|---:|---|---|---:|---:|---:|---:|
| 1 | Recursive / hybrid / rerank | chunk=256, RRF=60, rerank_top=10 | 0.5365 | 0.4850 | 0.4412 | 0.0764 s |
| 2 | Recursive / hybrid / rerank | chunk=256, RRF=60, rerank_top=5 | 0.5365 | 0.4850 | 0.4412 | 0.0650 s |
| 3 | Recursive / hybrid | chunk=256, RRF=30 | 0.5283 | 0.4615 | 0.4192 | 0.0075 s |
| 4 | Fixed / dense | chunk=512, overlap=32 | 0.5258 | 0.4483 | 0.4116 | 0.0063 s |
| 5 | Recursive / dense | chunk=256 | 0.5248 | 0.4445 | 0.4107 | 0.0063 s |
| 6 | Recursive / dense | chunk=256 | 0.5248 | 0.4445 | 0.4107 | 0.0076 s |
| 7 | Recursive / dense | chunk=512 | 0.5238 | 0.4463 | 0.4101 | 0.0069 s |
| 8 | Paragraph / dense | candidate=15 | 0.5238 | 0.4443 | 0.4096 | 0.0074 s |
| 9 | Paragraph / dense | candidate=20 | 0.5238 | 0.4443 | 0.4096 | 0.0068 s |
| 10 | Recursive / hybrid | chunk=256, RRF=60 | 0.5190 | 0.4565 | 0.4135 | 0.0077 s |
| 11 | Fixed / dense | chunk=256, overlap=32 | 0.5171 | 0.4413 | 0.4048 | 0.0066 s |
| 12 | Recursive / hybrid / multi-query | chunk=256, RRF=60, queries=3 | 0.5144 | 0.4381 | 0.4068 | 1.2640 s |
| 13 | Semantic / dense | percentile=90 | 0.5142 | 0.4504 | 0.4140 | 0.0072 s |
| 14 | Semantic / dense | percentile=90 | 0.5141 | 0.4533 | 0.4139 | 0.0074 s |
| 15 | Auto-merging / dense | merge=0.35 | 0.5132 | 0.4336 | 0.4049 | 0.0076 s |
| 16 | Parent-document / dense | parent=512 | 0.5115 | 0.4398 | 0.4093 | 0.0076 s |
| 17 | Parent-document / dense | parent=1024 | 0.5112 | 0.4435 | 0.4106 | 0.0073 s |
| 18 | Auto-merging / dense | merge=0.60 | 0.5075 | 0.4431 | 0.4068 | 0.0074 s |
| 19 | Recursive / hybrid / rewrite | chunk=256, RRF=60 | 0.5038 | 0.4276 | 0.3924 | 0.2889 s |
| 20 | Recursive / hybrid / multi-query | chunk=256, RRF=60, queries=2 | 0.4985 | 0.4484 | 0.3976 | 0.8147 s |
| 21 | Recursive / hybrid / hyde | chunk=256, RRF=60 | 0.4881 | 0.4159 | 0.3784 | 5.6490 s |
| 22 | Sentence window / dense | window=3 | 0.4668 | 0.3999 | 0.3690 | 0.0088 s |
| 23 | Sentence window / dense | window=1 | 0.4668 | 0.3993 | 0.3685 | 0.0089 s |
| 24 | Recursive / bm25 | chunk=256 | 0.4306 | 0.3673 | 0.3354 | 0.0003 s |
| 25 | Recursive / bm25 | chunk=256 | 0.4299 | 0.3692 | 0.3357 | 0.0002 s |

<details>
<summary>Exact run IDs, in leaderboard order</summary>

1. `recursive_hybrid_none_rerank_bd0feccaf82a`
2. `recursive_hybrid_none_rerank_941727ac31a6`
3. `recursive_hybrid_none_raw_579ce42d57d7`
4. `fixed_dense_none_raw_33d59b9de9aa`
5. `recursive_dense_none_raw_07bf33c4b819`
6. `recursive_dense_none_raw_04818043eb3f`
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

</details>

## Standouts and what the experiment actually shows

### 1. Hybrid + cross-encoder reranking leads, but its average advantage is modest

- **Winner:** Recall@5 **0.5365**, MRR **0.4850**, nDCG@5 **0.4412**, mean query latency **76.4 ms**.
- **Paragraph + dense baseline:** Recall@5 **0.5238**, MRR **0.4443**, mean latency **6.8 ms**.
- **Matched-question head-to-head:** winner improves Recall@5 on **49/253** questions, ties **162/253**, and performs worse on **42/253**. The small average gain hides meaningful per-question gains *and losses*.
- **Versus the same RRF=60 hybrid without reranking:** Recall@5 increases from **0.5190** to **0.5365**, while mean query latency increases from **7.7 ms** to **76.4 ms** (approximately **9.9×**).
- The two reranking variants (`rerank_top_n=5` and `10`) achieve identical retrieval scores. Increasing this parameter did not improve results here.

### 2. Auto-merging is competitive, not dominant

- Merge threshold **0.35**: Recall@5 **0.5132**, Precision@5 **0.1802**.
- Merge threshold **0.60**: Recall@5 **0.5075**, Precision@5 **0.1787**.
- Both auto-merging variants trail recursive hybrid on evidence recall. A parent-expanded context can nevertheless help synthesis; that hypothesis needs **matched generation evaluations**, which these exports do not contain for auto-merging.

### 3. Query rewriting, multi-query, and HyDE did not beat raw queries in this sweep

| Same recursive-hybrid family | Recall@5 | Mean latency |
|---|---:|---:|
| No transformation, RRF=60 | 0.5190 | 0.008 s |
| Rewrite | 0.5038 | 0.289 s |
| Multi-query (3 queries) | 0.5144 | 1.264 s |
| Multi-query (2 queries) | 0.4985 | 0.815 s |
| HyDE | 0.4881 | 5.649 s |

- Here, raw retrieval has the highest Recall@5 among the RRF=60/no-rerank transformation variants. Multi-query and HyDE add local LLM calls without improving average recall.
- **Latency outlier:** HyDE averages **5.65 s/query**, versus **0.0077 s/query** for raw hybrid retrieval. This is a very large cost for a lower score; benchmark conditions and cached-model state can affect timings.

### 4. BM25 is exceptionally fast, but it retrieves less gold evidence

- Best BM25-only Recall@5: **0.4306**; measured latency **0.28 ms/query**.
- The dashboard showing **0.000 s** is a **three-decimal rounding artifact**, not zero actual time. Compare the stored sub-millisecond values rather than the rounded KPI.
- Some BM25 runs emit top-K-clamping warnings on papers with fewer indexed nodes than requested. Those are warnings, not retrieval failures; candidate-pool sizes still matter in method comparisons.

### 5. Repeated score ties have concrete causes

- Two reranked hybrid runs have identical Recall, MRR, and nDCG despite different rerank candidate settings.
- Multiple dense variants also tie. The evaluation credits **unique gold paragraphs**, so different chunks can yield identical paragraph-level scores. Some parameter changes may not change the actual final top-5 results.
- An identical score is not proof of equivalent contexts, answer quality, or runtime.

## Answer quality — 30 generated answers

These rows come from **one** winning retrieval configuration, not one set of answers per retrieval method. The 30 saved rows span 11 papers and are an initial subset of the retrieval sample; results should not be interpreted as broad model-accuracy estimates.

| Metric | Valid / 30 | Mean | Interpretation |
|---|---:|---:|---|
| Groundedness | 30/30 | **0.523** | Local Ollama support-from-context proxy |
| Context Relevance | 30/30 | **0.690** | Local Ollama question/context proxy |
| Answer Relevance | 30/30 | **0.700** | Local Ollama answer/question proxy |
| Faithfulness | 27/30 | **0.783** | RAGAS answer claims supported by context |
| Context Precision | 30/30 | **0.553** | RAGAS relevance of ranked contexts against reference |
| Context Recall | 30/30 | **0.560** | RAGAS reference coverage from retrieved contexts |
| Answer Relevancy | 30/30 | **0.646** | RAGAS answer relevance to the question |
| Reference Token F1 | 30/30 | **0.122** | Lexical token F1 against a reference answer |

**Evaluation caveats.** The 3 missing faithfulness scores are failed judge outputs, **not scores of zero**. For the other RAGAS metrics all 30 rows scored successfully. Local proxy scores and RAGAS scores are different procedures and can disagree. Reference-token F1 is a lexical overlap measure, not a semantic correctness verdict; reference variants and long generated answers make it especially sensitive to wording. The RAGAS export selects a single reference string for some reference-dependent metrics even when QASPER provides multiple acceptable answers.

## Qualitative examples: where RAGxiv succeeds and fails

All examples below are real saved QASPER questions and Ollama answers, with the displayed scores aligned by row to `generation_scores.csv` and `ragas_scores.csv`. Answer excerpts are shortened for readability; inspect the CSV files for the full text.

### Example 3: Clear factual extraction — **success**

**Question:** which datasets did they experiment with?

**Reference answer (one of 2):** Europarl MultiUN

**Generated answer (excerpt):** “Based on the evidence provided, the datasets they experimented with are:  [2] Europarl BIBREF31 and [3] MultiUN BIBREF32. These two datasets were used to evaluate the cross-lingual pre-training based transfer approach on zero-shot translation tasks.”

**Scores:** Retrieval Recall@5 `1.00` · local groundedness `0.80` · RAGAS faithfulness `0.50` · RAGAS context recall `1.00` · answer relevancy `0.75` · reference token F1 `0.19`.

**Assessment:** The generated answer identifies both named datasets, Europarl and MultiUN, matching the reference. The extracted evidence is relevant, although the faithfulness judge scores only 0.50; a useful reminder that judge scores are estimates.

### Example 21: Correct method retrieval — **success**

**Question:** How do they calculate a static embedding for each word?

**Reference answer (one of 2):** They use the first principal component of a word's contextualized representation in a given layer as its static embedding.

**Generated answer (excerpt):** “To determine how static embeddings are calculated for each word, we need to use the provided evidence to deduce the process. Let's analyze the evidence step by step:  1. [1] In this evidence, we see that static embeddings are created by taking the first principal component (PC) of the contextualized representations of a word in a given layer.…”

**Scores:** Retrieval Recall@5 `1.00` · local groundedness `0.80` · RAGAS faithfulness `1.00` · RAGAS context recall `1.00` · answer relevancy `0.99` · reference token F1 `0.12`.

**Assessment:** The answer correctly explains that a word’s static embedding is the first principal component of its contextualized representations in a layer. High retrieval recall and judge faithfulness align on this example.

### Example 24: Opposite yes/no answer — **failure despite high faithfulness**

**Question:** Does BERT reach the best performance among all the algorithms compared?

**Reference answer (one of 1):** no

**Generated answer (excerpt):** “Based on the evidence provided, BERT does appear to reach the best performance among the compared algorithms. This can be inferred from the following points:  1. **[1]**: BERT outperforms other systems in terms of F1-score, with a significant margin. The difference in F1-score between BERT and the next best system is 1.9 points. Additionally,…”

**Scores:** Retrieval Recall@5 `0.25` · local groundedness `0.80` · RAGAS faithfulness `1.00` · RAGAS context recall `0.00` · answer relevancy `0.74` · reference token F1 `0.00`.

**Assessment:** The QASPER reference says “no”, but the generated answer claims BERT did achieve the best performance. This is an answer-correctness failure even though RAGAS faithfulness is 1.00, showing that support-from-retrieved-context is not equivalent to agreement with the reference.

### Example 10: Unanswered question — **retrieval/evidence gap**

**Question:** What crowdsourcing platform is used?

**Reference answer (one of 1):** They did not use any platform, instead they hired undergraduate students to do the annotation.

**Generated answer (excerpt):** “Based on the provided evidence, we cannot definitively determine which specific crowdsourcing platform is used. The evidence discusses the use of crowdsourcing for obtaining labeled data but does not specify the platform or platforms utilized. The evidence mentions Amazon Mechanical Turk [5] as a platform that has been used for obtaining…”

**Scores:** Retrieval Recall@5 `0.00` · local groundedness `0.00` · RAGAS faithfulness `0.75` · RAGAS context recall `0.00` · answer relevancy `0.00` · reference token F1 `0.06`.

**Assessment:** The reference says no crowdsourcing platform was used; undergraduate annotators were hired. The answer instead says the platform cannot be determined, with misleading mention of Mechanical Turk. Retrieval Recall@5 is zero.

### Example 12: Missing a requested number — **failure**

**Question:** On how many language pairs do they show that preordering assisting language sentences helps translation quality?

**Reference answer (one of 2):** 5

**Generated answer (excerpt):** “Based on the evidence provided, the research clearly indicates that preordering assisting language sentences to match the word order of the source language significantly improves translation quality, particularly in scenarios where there is no parallel corpus between the source and target language.   [1] and [3] both mention experiments showing…”

**Scores:** Retrieval Recall@5 `0.00` · local groundedness `0.00` · RAGAS faithfulness `0.50` · RAGAS context recall `0.00` · answer relevancy `0.00` · reference token F1 `0.06`.

**Assessment:** The question asks for the number of language pairs; the reference is “5”. The generated answer gives a generic qualitative explanation instead of the requested count.

### Example 2: Judge disagreement — **inspect manually**

**Question:** what are the pivot-based baselines?

**Reference answer (one of 2):** pivoting pivoting$_{\rm m}$

**Generated answer (excerpt):** “Based on the evidence provided, pivot-based baselines are a strategy to obtain a source-to-target model by introducing a pivot language. They are further divided into two methods: pivoting and pivot-synthetic. The pivoting method involves translating from the source language to the pivot language, and then translating from the pivot language to…”

**Scores:** Retrieval Recall@5 `0.75` · local groundedness `0.00` · RAGAS faithfulness `1.00` · RAGAS context recall `1.00` · answer relevancy `0.80` · reference token F1 `0.08`.

**Assessment:** The local proxy judge assigns zero across its three metrics, while RAGAS faithfulness, context precision, and recall are approximately 1. This is evaluator disagreement, not proof that either judge is right.

## Outliers, failures, and limitations

1. **Three incomplete local RAGAS faithfulness evaluations.** Saved rows 10, 14, and 19 (zero-based) have `IncompleteOutputException` / output truncated by `max_tokens`; faithfulness is missing for those rows. They should be retried with adequate output-token and context budgets before final publication of faithfulness statistics.
2. **Evidence mapping coverage:** 21 of 274 sampled questions are not mapped to scored evidence. These are excluded from averaged retrieval metrics, so report coverage alongside recall.
3. **Large question-level variance:** The best method loses to the paragraph baseline on 42/253 questions despite winning on average. Examine high-impact misses rather than selecting a winner from a single average.
4. **Reference correctness is not the same as groundedness.** The BERT yes/no example has faithfulness 1.0 but contradicts the gold answer. A reference-aware factual-correctness or exact-answer validation metric is needed.
5. **Short-answer/reference mismatch:** Long explanatory answers can have low token F1 despite correctly containing short named entities or technical terms. A robust evaluation should compare semantic equivalence and normalized final answers.
6. **Single-configuration generation:** The 30 answers are generated using a leading reranked hybrid configuration; this export does not provide matched answer generation for auto-merging, HyDE, BM25 or rewrite. Therefore, no answer-quality ranking among retrieval techniques can be claimed.
7. **Speed measurements:** the recorded query latency is reported per experiment. The dashboard’s 0.000 s is rounding of fast BM25 measurements; timings are not a cross-hardware benchmark and do not necessarily include all model loading/index-building costs.
8. **Experiment variants are not a fully controlled ablation:** e.g. RRF=30 also changes candidate counts in the export. Only compare parameter effects when other settings match.

## Recommended next experiments

- **Hold the evaluation set fixed:** generate answers on the same 100+ questions for recursive-hybrid-rerank, recursive-hybrid-raw, paragraph-dense, and auto-merging; compare cost, correctness, and citations rather than recall alone.
- **Add reference-aware correctness:** short-answer normalized exact match (when applicable), semantic factual correctness, and explicit yes/no accuracy; use multiple QASPER reference variants instead of arbitrarily choosing the first one.
- **Repair failed judge rows:** cache scores by question/configuration/metric and retry only the 3 missing faithfulness scores; validate output token/context limits first.
- **Investigate per-question regressions:** inspect the 42 questions where the winner loses to paragraph-dense, and analyze query types (numeric, yes/no, entity/list, multi-hop).
- **Compare retrieval latency fairly:** break down indexing, embedding, sparse/dense retrieval, transformation LLM time, reranking, and answer generation into separate measurements.

## Running locally

Python dependencies and the exact pipeline are maintained in the accompanying notebook and `requirements.txt`. Example setup:

```bash
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

Run the RAGxiv notebook to build/load indexes, execute the retrieval sweep, generate answers, and run the optional local RAGAS judge. The exported artifacts used here are:

```text
artifacts/
├── runs/
│   ├── <run_id>_summary.json   # configuration, retrieval metrics, timings
│   └── <run_id>.json           # question-level evidence and contexts
├── generation_scores.csv      # questions, references, answers, local judge scores
├── ragas_scores.csv           # reference-aware / judge-based evaluation outputs
└── sweep_failures.json        # [] in the supplied export
```

> **Publishing note:** Paper passages and generated answers in the artifacts can be large. Review licensing, dataset attribution, file sizes and any project-specific privacy requirements before committing raw `artifacts/` to a public repository. This README summarizes them without embedding the full QASPER dataset.

## Bottom line

**Recursive hybrid with reranking is the leading retrieval strategy in this snapshot** (Recall@5 0.5365), but the improvement over paragraph-dense is only +0.0127 on average, with substantial question-level tradeoffs and higher latency. Auto-merging remains competitive; rewrite, multi-query and HyDE do not outperform raw hybrid retrieval here. The 30-answer evaluation demonstrates a key limitation: **an answer can sound well-supported and still disagree with the reference answer**. Future work should prioritize controlled, matched answer-generation comparisons and reference-aware correctness evaluation.

