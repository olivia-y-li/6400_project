# hotpotqa retrieval results (k=5, min_overlap=0.5)

Mean [95% bootstrap CI over questions]. Report the eval rows; dev is for debugging only.

| system/split | Recall@5 | Complete Recall@5 | MRR@5 |
|---|---|---|---|
| bm25/dev | 0.607 [0.537, 0.677] | 0.240 [0.120, 0.360] | 0.841 [0.761, 0.920] |
| bm25/eval | 0.668 [0.640, 0.697] | 0.416 [0.370, 0.462] | 0.772 [0.739, 0.801] |
| dense/dev | 0.713 [0.640, 0.783] | 0.440 [0.320, 0.580] | 0.888 [0.815, 0.952] |
| dense/eval | 0.720 [0.693, 0.746] | 0.488 [0.444, 0.532] | 0.855 [0.829, 0.881] |
| hybrid/dev | 0.683 [0.617, 0.750] | 0.360 [0.220, 0.480] | 0.883 [0.822, 0.945] |
| hybrid/eval | 0.740 [0.713, 0.765] | 0.506 [0.460, 0.548] | 0.848 [0.822, 0.872] |
| hybrid_rerank/dev | 0.763 [0.693, 0.827] | 0.520 [0.380, 0.660] | 0.975 [0.935, 1.000] |
| hybrid_rerank/eval | 0.831 [0.809, 0.852] | 0.662 [0.620, 0.704] | 0.954 [0.938, 0.968] |

## By question type (eval, Complete Recall)

| system | bridge | comparison |
|---|---|---|
| bm25 | 0.411 (n=409) | 0.440 (n=91) |
| dense | 0.438 (n=409) | 0.714 (n=91) |
| hybrid | 0.462 (n=409) | 0.703 (n=91) |
| hybrid_rerank | 0.606 (n=409) | 0.912 (n=91) |

## Failure breakdown (eval)

| system | complete | partial | miss |
|---|---|---|---|
| bm25 | 208 | 246 | 46 |
| dense | 244 | 229 | 27 |
| hybrid | 253 | 229 | 18 |
| hybrid_rerank | 331 | 165 | 4 |

## Paired bootstrap between consecutive systems (eval)

Difference in means (second minus first) [95% CI], two-sided p.

| comparison | metric | difference | p |
|---|---|---|---|
| bm25 -> dense | recall | +0.052 [+0.020, +0.085] | 0.0010 |
| bm25 -> dense | complete_recall | +0.072 [+0.020, +0.124] | 0.0054 |
| bm25 -> dense | mrr | +0.083 [+0.050, +0.117] | 0.0000 |
| dense -> hybrid | recall | +0.020 [-0.001, +0.041] | 0.0660 |
| dense -> hybrid | complete_recall | +0.018 [-0.018, +0.056] | 0.3636 |
| dense -> hybrid | mrr | -0.007 [-0.029, +0.015] | 0.5400 |
| hybrid -> hybrid_rerank | recall | +0.091 [+0.069, +0.114] | 0.0000 |
| hybrid -> hybrid_rerank | complete_recall | +0.156 [+0.116, +0.196] | 0.0000 |
| hybrid -> hybrid_rerank | mrr | +0.106 [+0.082, +0.131] | 0.0000 |

Questions where one system has the higher Complete Recall:

- bm25 -> dense: second better 105, first better 69, tie 326
- dense -> hybrid: second better 50, first better 41, tie 409
- hybrid -> hybrid_rerank: second better 96, first better 18, tie 386
