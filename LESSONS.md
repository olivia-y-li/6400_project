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
