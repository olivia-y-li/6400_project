"""Run every system on dev and eval and write one comparison report.

python -m rag.report --data data/processed/hotpotqa --out data/results/hotpotqa
Dev is for debugging only; report the eval rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import MIN_OVERLAP
from .evaluate import score_split, summarize
from .metrics import METRICS, paired_bootstrap
from .retrieval import SYSTEMS, build_systems
from .schema import Chunk, Question, read_jsonl


def wins(a: list[dict], b: list[dict], metric: str = "complete_recall") -> dict[str, int]:
    """Questions where a beats b / b beats a / tie, on one metric."""
    return {
        "a_better": sum(x[metric] > y[metric] for x, y in zip(a, b)),
        "b_better": sum(y[metric] > x[metric] for x, y in zip(a, b)),
        "tie": sum(x[metric] == y[metric] for x, y in zip(a, b)),
    }


def run(data_dir: str | Path, out_dir: str | Path, k: int = 5, min_overlap: float = MIN_OVERLAP,
        systems: tuple[str, ...] = SYSTEMS, title_ablation: bool = False) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    chunks = list(read_jsonl(data_dir / "chunks.jsonl", Chunk))
    by_id = {c.chunk_id: c for c in chunks}
    splits = {s: list(read_jsonl(data_dir / f"questions_{s}.jsonl", Question))
              for s in ("dev", "eval")}

    retrievers = dict(build_systems(chunks, systems))
    if title_ablation:  # not one of the five systems; chunk text is prefixed with the title
        retrievers.update({f"{n}+title": r for n, r in
                           build_systems(chunks, ["bm25", "dense"], prepend_title=True).items()})

    results, per_q_all = {}, {}
    for name, r in retrievers.items():
        for split, qs in splits.items():
            per_q, details = score_split(r, qs, by_id, k, min_overlap)
            results[f"{name}/{split}"] = summarize(per_q, qs)
            per_q_all[f"{name}/{split}"] = (per_q, details)

    # Consecutive systems, in the order given: each is compared with the one before it.
    comparison = {}
    for first, second in zip(systems, systems[1:]):
        a, b = per_q_all[f"{first}/eval"][0], per_q_all[f"{second}/eval"][0]
        comparison[f"{first} -> {second}"] = {
            "paired_bootstrap": {m: paired_bootstrap(a, b, m) for m in METRICS},
            "complete_recall_wins": wins(a, b),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(
        {"k": k, "min_overlap": min_overlap, "results": results, "comparison": comparison,
         "per_question": {key: d for key, (_, d) in per_q_all.items()}}, indent=1))
    (out_dir / "report.md").write_text(to_markdown(results, comparison, k, min_overlap, data_dir.name))
    return {"results": results, "comparison": comparison}


def to_markdown(results: dict, comparison: dict, k: int, min_overlap: float,
                benchmark: str = "") -> str:
    def row(key: str) -> str:
        r = results[key]
        cell = lambda m: f"{r[m]:.3f} [{r['ci95'][m][0]:.3f}, {r['ci95'][m][1]:.3f}]"
        return f"| {key} | {cell('recall')} | {cell('complete_recall')} | {cell('mrr')} |"

    head = f"| system/split | Recall@{k} | Complete Recall@{k} | MRR@{k} |\n|---|---|---|---|\n"
    lines = [f"# {benchmark} retrieval results (k={k}, min_overlap={min_overlap})\n",
             "Mean [95% bootstrap CI over questions]. Report the eval rows; dev is for debugging only.\n",
             head + "\n".join(row(k_) for k_ in results)]
    qtypes = sorted({t for r in results.values() for t in r["by_qtype"]})
    if qtypes:
        lines += ["\n## By question type (eval, Complete Recall)\n",
                  "| system | " + " | ".join(qtypes) + " |\n|---|" + "---|" * len(qtypes)]
        for key, r in results.items():
            if key.endswith("/eval"):
                lines.append(f"| {key[:-5]} | " + " | ".join(
                    f"{r['by_qtype'][t]['complete_recall']:.3f} (n={r['by_qtype'][t]['n']})"
                    for t in qtypes) + " |")
    lines += ["\n## Failure breakdown (eval)\n", "| system | complete | partial | miss |\n|---|---|---|---|"]
    for key, r in results.items():
        if key.endswith("/eval"):
            f = r["failures"]
            lines.append(f"| {key[:-5]} | {f['complete']} | {f['partial']} | {f['miss']} |")
    if comparison:
        lines += ["\n## Paired bootstrap between consecutive systems (eval)\n",
                  "Difference in means (second minus first) [95% CI], two-sided p.\n",
                  "| comparison | metric | difference | p |\n|---|---|---|---|"]
        for pair, c in comparison.items():
            for m, v in c["paired_bootstrap"].items():
                lines.append(f"| {pair} | {m} | {v['diff']:+.3f} [{v['ci95_low']:+.3f}, "
                             f"{v['ci95_high']:+.3f}] | {v['p']:.4f} |")
        lines += ["\nQuestions where one system has the higher Complete Recall:\n"]
        for pair, c in comparison.items():
            w = c["complete_recall_wins"]
            lines.append(f"- {pair}: second better {w['b_better']}, first better {w['a_better']}, tie {w['tie']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag.report")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min-overlap", type=float, default=MIN_OVERLAP)
    p.add_argument("--title-ablation", action="store_true",
                   help="also run bm25/dense with the page title prefixed (not one of the five systems)")
    a = p.parse_args(argv)
    run(a.data, a.out, a.k, a.min_overlap, title_ablation=a.title_ablation)
    print((Path(a.out) / "report.md").read_text())


if __name__ == "__main__":
    main()
