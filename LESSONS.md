# Lessons log

Dated entries for anything that failed or changed design. The report's Lessons Learned
section is written from this file.

## 2026-11-04: FAISS replaced by a NumPy matrix for dense retrieval
FAISS and PyTorch crashed the interpreter on macOS (native-library conflict), and PyTorch's
Apple MPS backend crashed during MiniLM inference. Dense retrieval now keeps the L2-normalized
embeddings in a NumPy matrix and ranks by dot product on CPU. This is exact cosine search, so
the results are identical to a flat inner-product FAISS index (verified: milestone numbers
reproduce). Commit dcec45f.

## 2026-11-23: QASPER loading script is blocked
`allenai/qasper` ships a loading script, and `datasets` 5.x refuses to run scripts
("Dataset scripts are no longer supported"). We load the Hub's auto-converted parquet files
instead (`revision="refs/convert/parquet"`, `qasper/test/0000.parquet`). Same 416 test papers
and 1,451 questions.

## 2026-11-23: QASPER matching rule for evidence longer than a chunk
Evidence units are paragraphs, and 7.4% of paragraphs (10.5% of gold ones) are longer than the
1,000-character window; the longest is about 19,000 characters. Under "a chunk must cover
>= 50% of the paragraph", a paragraph over 2,000 characters could never be retrieved by any
chunk. Decision: a chunk's coverage of a unit is its overlap in characters divided by
min(unit length, chunk size), and a unit counts as retrieved when that is >= 0.5
(min_overlap). For units that fit in one window (all HotpotQA sentences but one) this is
identical to the plain fraction, so HotpotQA is unchanged: the rebuilt chunks differ from the
old ones in one non-gold paragraph, and the baseline numbers reproduce exactly.
Effect: eval questions where every gold unit is retrievable: 100%; where the evidence also
fits in the top 5 chunks: 99.8%.

## 2026-11-23: QASPER evidence that is only a section heading
218 of 5,285 non-figure evidence strings are section titles such as
"Experiments ::: Automatic Evaluation Metrics", not paragraph text. They cannot be mapped to a
paragraph without inventing evidence, so they are dropped and counted separately
(`section_heading_only`). No evidence string failed to match for any other reason, and no
fuzzy matches were needed (5,063 exact, 4 substring).

## 2026-11-23: QASPER annotators
Each usable annotator (not unanswerable, at least one mapped paragraph) gives one gold evidence
set and one gold answer. Retrieval metrics score a question against its best evidence set;
`gold_chunk_ids` (used for MRR) is the union over sets. 1,309 of 1,451 questions have at least
one usable annotator; eval questions average 2.22 evidence sets. Answer F1 should take the max
over `gold_answers` for consistency.

## 2026-11-23: QASPER pooled-corpus retrieval is much harder than HotpotQA
Baseline Recall@5 on the pooled 313-paper corpus is 0.18 (BM25) and 0.15 (dense), against
0.67 / 0.72 on HotpotQA. To check this was the benchmark and not a labeling bug, BM25 was run
restricted to each question's own paper: Recall@5 rises to 0.50 and Complete Recall@5 to 0.46.
QASPER questions often do not name their paper ("How big is the dataset?"), so a pooled corpus
leaves them ambiguous. Adding the paper title to the chunk text did not help either retriever.

## 2026-11-25: Hybrid (RRF) and cross-encoder rerank
Hybrid takes the top 50 from BM25 and from dense and fuses them with RRF (constant 60);
Hybrid + Rerank rescores the fused top 50 with `cross-encoder/ms-marco-MiniLM-L-6-v2`, run on
CPU like the dense model, and keeps 5. The cross-encoder reads the chunk text as stored, with
no title prefix, so all systems see identical chunks. Pool size and RRF constant were fixed in
advance and not tuned.
- HotpotQA (eval, Complete Recall@5): BM25 0.416, dense 0.488, hybrid 0.506, hybrid + rerank
  0.662. Hybrid over dense is NOT significant (+0.018, p = 0.36); rerank over hybrid is
  (+0.156, p < 0.001) and fixes bridge questions most (0.462 -> 0.606; comparison 0.703 -> 0.912).
- QASPER (eval, Complete Recall@5): BM25 0.168, dense 0.134, hybrid 0.204, hybrid + rerank
  0.250. Here hybrid over dense is significant (+0.070, p < 0.001), and dense is slightly worse
  than BM25 (-0.034, p = 0.044).
- Cost: scoring 50 chunk pairs per question on CPU made the full report (both splits, four
  systems) take about 7 minutes per benchmark. Latency per stage is logged separately in the
  latency/cost step.

## 2026-11-25: Generation cannot use temperature 0
The design calls for Claude Haiku 4.5 at temperature 0. The Anthropic SDK (2.14) has no
`temperature` argument, and sending one in the raw request body is rejected by the API
(`temperature: Extra inputs are not permitted`). Sampling therefore uses the API default.
Reproducibility comes from the on-disk response cache (keyed by a hash of model, prompt and
tool schema): the first answer for a prompt is stored and every rerun replays it exactly.
Consequence for the report: results are reproducible from the cache, but a fresh run with an
empty cache could differ slightly. Decomposition uses the same client and cache.

## 2026-11-25: Generation prompt revised once on the HotpotQA dev set
First prompt (hybrid + rerank, 50 dev questions): EM 0.34, F1 0.52. Six answers said the
passages did not contain the answer, and several added qualifiers or explanations. Revised the
prompt to ask for only the answer span in the passages' own wording, with no parentheticals,
and never to refuse. Same dev questions: EM 0.42, F1 0.54, five non-answers. Further tuning
was stopped, since 50 questions would be overfit. The final prompt is shared by all systems
and both benchmarks. Remaining misses are mostly missing evidence or metric noise
("Switzerland" vs "Swiss", "Dogs" vs "dog").

## 2026-11-25: An API spend limit is not a credit balance
The first API call failed with "credit balance is too low": the monthly spend limit only caps
spending, and prepaid credits had to be added separately.

## 2026-11-25: QASPER dev answers cite gold evidence far less often
On the 50 QASPER dev questions (hybrid + rerank), 22% of answers cite a gold chunk versus 98%
on HotpotQA, and answer F1 is 0.33 versus 0.54. Citations track retrieval: the model cites what
it was given, and the pooled corpus often hides the right paragraph.

## 2026-11-25: First eval run left some answers uncited; prompt revised again
The first 4-system eval (500 questions per benchmark) cost $9.16, but 2-12% of answers had no
valid citation (HotpotQA BM25 92.6% cited, QASPER dense 87.4%). Almost all uncited answers were
refusals ("The passages do not contain...", "<UNKNOWN>") that scored about 0 F1, and they were
concentrated in the weaker retrievers where the evidence was missing. The design requires every
answer to name its supporting chunks, so the prompt now says to always cite at least one
passage (the closest, if unsure), and the output schema requires `minItems: 1`. On 50 dev
questions each, valid-citation rate went to 100% for BM25 on HotpotQA and dense on QASPER, so
the fix works; the model still writes a refusal as the answer text in about 20% of those weak
cases, but it now cites a passage with it. The old-prompt results are kept under `qa_old_prompt/`
and the whole eval was rerun with the new prompt, since all systems must share one prompt.

## 2026-11-25: Hard spend cap inside the LLM wrapper
Prepaid credit is limited, so `LLM` now enforces a lifetime cap ($33 of $35 credit) itself:
before every live call it reserves a worst-case $0.01 under a lock, and refuses once total spend
(recomputed from every token count in the cache, plus this session) would pass the cap. Cached
calls are free and never blocked. Per-run `--max-cost` flags sit on top of this. `python -m
rag.llm` prints the running total. One gap: a billed response with no usable tool call is not
cached, so it is not in the ledger; the $2 margin covers that.

## 2026-11-25: Decomposition is a null result for retrieval, and a small gain for answers
Final eval, Hybrid + Rerank -> + Decomposition (500 questions each):
- HotpotQA: Complete Recall@5 0.662 -> 0.666 (p = 0.71), Recall@5 0.831 -> 0.833, MRR unchanged.
  Decomposition ran on 82.8% of questions (1.69 LLM calls per question) but changed the top-5
  chunk set for only 66 of 500; on the other 434 the retrieved chunks and the answers are
  identical. Answer F1 0.644 -> 0.657 (paired +0.013, p = 0.021); the whole gain is on the 66
  changed questions (F1 0.537 -> 0.631). Complete Recall was better on 9 questions and worse on 7.
- QASPER: nothing moves (Complete Recall 0.250 -> 0.248, F1 0.271 -> 0.274, both p > 0.4).
  Decomposition ran on only 15.4% of questions, and where it ran it often introduced a wrong
  bridge entity (for example "they" resolved to "humans" or to an unrelated clinical system).
- Likely reason retrieval barely moves: the final rerank scores against the ORIGINAL question,
  so sub-question candidates only displace the plain hybrid top-5 when they outscore it.
  The `--include-original` ablation (also pool the undecomposed question) was not run.
- Decomposition prompt revised once on dev: later sub-questions used "these actors" instead of
  the {n} placeholder, and QASPER questions with pronouns were split into "what does 'they'
  refer to" sub-questions. After the change QASPER dev decomposition fell from 32% to 6% of
  questions and 35 of 44 decomposed HotpotQA dev questions had a bridge answer filled in.
- Design choice recorded: intermediate answers are produced from hybrid + rerank on the
  sub-question (top 5), not from plain hybrid.

## 2026-11-25: Total API spend
All generation and decomposition runs cost $21.56 in total (ledger from `python -m rag.llm`),
of which $9.16 went on a first 4-system eval that had to be rerun for the citation fix. Unused
results from that run are kept under `results/<benchmark>/qa_old_prompt/`.
