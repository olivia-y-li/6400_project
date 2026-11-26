# Handoff: where the project stands

Written 2026-11-25. All five systems have run on both benchmarks (500 eval questions each) and
everything below the experiments is analysis, tables and writing. **No further API spend is
needed** for the remaining work. The report is due **Dec 7**; freeze experiments by **Dec 4-5**.

## 1. Headline results (eval split, k = 5)

Full tables with 95% CIs and paired tests: `results/SUMMARY.md` (regenerate with
`python -m rag.summary --results results`).

| system | HotpotQA Complete Recall@5 | HotpotQA answer F1 | QASPER Complete Recall@5 | QASPER answer F1 |
|---|---|---|---|---|
| BM25 | 0.416 | 0.561 | 0.168 | 0.237 |
| Dense | 0.488 | 0.554 | 0.134 | 0.220 |
| Hybrid (RRF) | 0.506 | 0.584 | 0.204 | 0.265 |
| Hybrid + Rerank | **0.662** | 0.644 | **0.250** | 0.271 |
| + Decomposition | **0.666** | **0.657** | 0.248 | **0.274** |

What the data say (all backed by paired bootstrap tests in `SUMMARY.md`):
- **Cross-encoder rerank is the one large win** on both benchmarks (HotpotQA +0.156 Complete
  Recall, p < 0.001; QASPER +0.046, p = 0.002).
- **Hybrid over dense**: significant on QASPER (+0.070), not on HotpotQA (+0.018, p = 0.38).
- **Dense vs BM25 flips by benchmark**: dense wins on HotpotQA, BM25 slightly wins on QASPER.
- **Decomposition is a null result for retrieval** on both benchmarks. It changes the top-5 for
  only 66/500 HotpotQA questions; answer F1 rises +0.013 (p = 0.02), all from those 66.
  On QASPER it rarely fires and often invents a wrong bridge entity.
- **QASPER is hard because the corpus is pooled**: restricting BM25 to a question's own paper
  lifts Recall@5 from 0.18 to 0.50 (see `LESSONS.md`). Only about 20% of QASPER answers cite a
  gold chunk even for the best system.

## 2. What is in the repo

| path | what |
|---|---|
| `src/rag/schema.py` | dataclasses (Document, Section, EvidenceUnit, Chunk, Question) + JSONL io |
| `parse_hotpot.py`, `parse_qasper.py` | raw HuggingFace rows -> Documents + Questions |
| `chunker.py` | 1000/100 char windows, `unit_coverage`, gold-chunk labeling |
| `build.py` | `python -m rag.build --benchmark hotpotqa\|qasper --out data/processed/<name>` |
| `retrieval.py` | BM25, dense (MiniLM, NumPy), Hybrid (RRF), Rerank (cross-encoder) |
| `decompose.py`, `decomp_run.py` | the full system (decomposition + hybrid + rerank) |
| `llm.py`, `generate.py` | Claude Haiku 4.5 wrapper (disk cache, cost, **spend cap**) and the answer prompt |
| `qa.py`, `qa_metrics.py` | cited answers per system/split; EM and token F1 |
| `metrics.py`, `evaluate.py`, `report.py`, `summary.py` | retrieval metrics, bootstrap CIs, paired tests, tables |
| `results/` | all final outputs, machine-readable, plus `environment.json` |
| `LESSONS.md` | dated log of every failure or design change (the Lessons Learned source) |

`results/<benchmark>/`: `report.json` and `decomp_eval.json` hold per-question retrieved chunk
IDs and metrics for all five systems; `qa/<system>_eval.jsonl` holds per-question answers,
citations (with document titles), tokens, cost and latency; `decomp_eval.json` also holds each
question's decomposition trace (sub-questions, filled-in bridge answers, per-stage timing).
`qa_old_prompt/` is the first eval, superseded by the citation fix; keep it only for the
lessons-learned story.

## 3. Setting up on a new machine

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,retrieval,llm]"
```
Then unzip `handoff_bundle.zip` (shared separately, not in git) into the project root. It holds
`data/llm_cache/` (every model answer) and `data/processed/` (the built corpora). With these in
place, everything below runs **without an API key**, because cached requests are replayed and the
Anthropic client is only created on a cache miss. Without the processed data, rebuild with
`rag.build` (deterministic; needs the HuggingFace download).

`pytest` should show 68 passing.

## 4. What is left

Experiments, in priority order (all free if the cache is in place):

1. **k-sweep (k = 1, 3, 5, 10) and plot.** Saved rankings only hold the top 5. Rerun at k = 10:
   `python -m rag.report --data data/processed/<b> --k 10 --out <dir>` for systems 1-4 and
   `python -m rag.decomp_run ... --k 10 --splits eval` for system 5 (its LLM calls replay from the
   cache), then score at each k with `metrics.score_question(q, ranked, chunks, k)`. Rankings at
   smaller k are prefixes of the k = 10 list.
2. **Per-stage latency table.** Report needs retrieval / fusion / rerank / decomposition /
   generation, with mean, p50, p95, plus LLM calls, tokens and dollars per question (tokens and
   dollars are already in the logs). The existing runs used 4-8 parallel workers, which inflates
   local timings, so do a **sequential** timing pass over a sample (a cache replay gives LLM stage
   latencies as originally measured). BM25, dense query encoding, fusion and rerank are not yet
   timed separately; they need small timers in `retrieval.py`.
3. **Error analysis.** Questions the full system still misses; wrong-bridge-entity cases from
   the decomposition traces; where BM25 beats dense and the reverse. All inputs are in `results/`.
4. **Ablations (retrieval-only, free):** title prefix (`rag.report --title-ablation`) and
   decomposition with the original question also pooled (`rag.decomp_run --include-original`; its
   answer run would cost about $1-2 and is optional).

Report materials:

5. **3-4 provenance examples** (question, gold, answer, cited chunk IDs with titles and text,
   whether they hit gold). Include at least one decomposition success and one failure. The
   66 HotpotQA questions where decomposition changed the chunk set are the place to look.
6. **Architecture diagram sources and pseudocode** (RRF, decomposition loop, rerank).
7. **From-scratch vs reused list.** Reused: `rank_bm25`, `sentence-transformers` models,
   `datasets`, the Anthropic API. Written here: parsing, evidence mapping, chunker, coverage-based
   gold labeling, RRF, the decomposition algorithm, metrics, bootstrap tests, spend cap.
8. **Check the QA scorers** (`qa_metrics.py`) against the official HotpotQA and QASPER evaluation
   scripts. They were written from memory of the SQuAD-style logic; numbers should be verified.

Housekeeping:

9. **Merge the PRs in order**: `qasper-pipeline`, `hybrid-rerank`, `generation`, `decomposition`
   (each stacked on the one before). `main` has none of this yet.

## 5. Decisions and gotchas the report must reflect

- **No temperature control.** The API rejects a temperature parameter for this model, so the
  design's "temperature 0" could not be followed. Reproducibility comes from the response cache.
  Do not delete `data/llm_cache`; a fresh run could give slightly different answers.
- **HotpotQA uses only `hard` questions** (matches the original loader; the spec was silent).
- **Matching rule:** a gold unit is retrieved if a chunk covers at least 50% of
  `min(unit length, chunk size)`. For HotpotQA sentences (all under 1000 chars) this is the
  plain 50% rule; the cap lets QASPER paragraphs longer than a chunk be retrievable at all.
- **QASPER evidence**: figure/table evidence dropped; 218 evidence strings that were only a
  section title also dropped. Each usable annotator is one gold evidence set; retrieval scores a
  question against its best set, and `gold_chunk_ids` (for MRR) is the union over sets.
- **Dense retrieval uses NumPy, not FAISS**, because FAISS/PyTorch crashed on macOS (also pinned
  to CPU because Apple MPS crashed). Results are mathematically identical.
- **Dev sets (50 questions) were used for prompt work.** Never tune on the eval set, and do not
  report dev numbers. The generation prompt was revised on dev for answer style, and the
  decomposition prompt once on dev. One exception to disclose: the citation requirement was
  triggered by the uncited-answer rate seen in the FIRST eval run (2-12% of answers had no
  citation). That change was a requirement fix, not accuracy tuning (it was verified on dev, then
  all eight evals were rerun), and the first run's results are kept in `qa_old_prompt/`.
- **The title-prefix variants are an ablation**, not one of the five systems.
- **Latency measured under 4-8 workers is inflated** (see item 2 above).
- **Spend:** live API spend so far is **$21.56** (check with `python -m rag.llm`). The LLM
  wrapper enforces a $33 lifetime cap, tuned to the original owner's prepaid credit. Anyone
  running live calls with their own key should set `LIFETIME_CAP_USD` in `llm.py` to their own
  limit. The key lives in a gitignored `.env` (`ANTHROPIC_API_KEY=...`) and must never be
  committed or shared.
