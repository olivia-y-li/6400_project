# qasper retrieval results (k=5, min_overlap=0.5)

Mean [95% bootstrap CI over questions]. Report the eval rows; dev is for debugging only.

| system/split | Recall@5 | Complete Recall@5 | MRR@5 |
|---|---|---|---|
| bm25/dev | 0.192 [0.090, 0.306] | 0.180 [0.080, 0.300] | 0.155 [0.068, 0.252] |
| bm25/eval | 0.179 [0.148, 0.214] | 0.168 [0.136, 0.202] | 0.105 [0.085, 0.129] |
| dense/dev | 0.190 [0.080, 0.300] | 0.180 [0.080, 0.300] | 0.102 [0.043, 0.175] |
| dense/eval | 0.150 [0.119, 0.180] | 0.134 [0.104, 0.166] | 0.097 [0.074, 0.119] |
| hybrid/dev | 0.240 [0.120, 0.360] | 0.240 [0.120, 0.360] | 0.165 [0.080, 0.255] |
| hybrid/eval | 0.217 [0.182, 0.254] | 0.204 [0.168, 0.242] | 0.137 [0.112, 0.164] |
| hybrid_rerank/dev | 0.310 [0.180, 0.430] | 0.300 [0.180, 0.420] | 0.221 [0.125, 0.329] |
| hybrid_rerank/eval | 0.268 [0.232, 0.307] | 0.250 [0.212, 0.290] | 0.173 [0.146, 0.202] |

## Failure breakdown (eval)

| system | complete | partial | miss |
|---|---|---|---|
| bm25 | 84 | 13 | 403 |
| dense | 67 | 16 | 417 |
| hybrid | 102 | 14 | 384 |
| hybrid_rerank | 125 | 20 | 355 |

## Paired bootstrap between consecutive systems (eval)

Difference in means (second minus first) [95% CI], two-sided p.

| comparison | metric | difference | p |
|---|---|---|---|
| bm25 -> dense | recall | -0.030 [-0.062, +0.003] | 0.0708 |
| bm25 -> dense | complete_recall | -0.034 [-0.066, -0.002] | 0.0440 |
| bm25 -> dense | mrr | -0.009 [-0.033, +0.016] | 0.4956 |
| dense -> hybrid | recall | +0.068 [+0.040, +0.096] | 0.0000 |
| dense -> hybrid | complete_recall | +0.070 [+0.042, +0.098] | 0.0000 |
| dense -> hybrid | mrr | +0.040 [+0.021, +0.060] | 0.0000 |
| hybrid -> hybrid_rerank | recall | +0.051 [+0.021, +0.081] | 0.0006 |
| hybrid -> hybrid_rerank | complete_recall | +0.046 [+0.016, +0.076] | 0.0016 |
| hybrid -> hybrid_rerank | mrr | +0.036 [+0.015, +0.057] | 0.0008 |

Questions where one system has the higher Complete Recall:

- bm25 -> dense: second better 25, first better 42, tie 433
- dense -> hybrid: second better 43, first better 8, tie 449
- hybrid -> hybrid_rerank: second better 40, first better 17, tie 443
