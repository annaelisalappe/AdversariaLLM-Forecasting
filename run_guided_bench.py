#!/usr/bin/env python3
"""
Run GuidedBench evaluation on AdversariaLLM run.json output files.

Usage:
    python run_guided_bench.py path/to/run.json [path/to/another/run.json ...]

The scores are written back into each run.json under step["scores"]["guidedbench_gemma4"].
"""
import argparse
import csv
import json
import os
import sys
import time

GUIDEDBENCH_ROOT     = "/nfs/homedirs/lapan/GuidedBench"
GUIDEDBENCH_EVAL_DIR = os.path.join(GUIDEDBENCH_ROOT, "Evaluation")
GUIDEDBENCH_CSV      = os.path.join(GUIDEDBENCH_ROOT, "Dataset", "GuidedBench.csv")
GUIDEDBENCH_DB       = os.path.join(GUIDEDBENCH_ROOT, "Dataset", "record.db")

sys.path.insert(0, GUIDEDBENCH_EVAL_DIR)

from evaluator import Evaluator_GuidedBench


def ensure_db(csv_path: str, db_path: str) -> None:
    """Build the SQLite DB from the CSV if it doesn't already exist."""
    if os.path.exists(db_path):
        return
    from utils.database import DataBase
    engine = DataBase(db_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with engine.engine.atomic():
        for row in rows:
            engine.GuidedBench.create(
                topic=row["topic"],
                question=row["question"],
                e_d1=row["entity_d1"],
                e_d2=row["entity_d2"],
                e_d3=row["entity_d3"],
                f_d1=row["function_d1"],
                f_d2=row["function_d2"],
                f_d3=row["function_d3"],
                e_e1=row["entity_e1"],
                e_e2=row["entity_e2"],
                e_e3=row["entity_e3"],
                f_e1=row["function_e1"],
                f_e2=row["function_e2"],
                f_e3=row["function_e3"],
                target=row["target"],
            )
    print(f"Built {db_path} with {len(rows)} records.")


def load_run(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not load {path}: {e}")
        return None


def save(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def judge_run(data: dict, evaluator: Evaluator_GuidedBench, score_key: str, path: str, batch_size: int = 8) -> dict:
    n = len(data["runs"])
    for i, subrun in enumerate(data["runs"]):
        behaviour_id = data["config"]["dataset_params"]["idx"]
        print(f"Commencing judge run for behaviour ID {behaviour_id}.", flush=True)
        question = subrun["original_prompt"][0]["content"]
        print(question, flush=True)
        for j, step in enumerate(subrun["steps"]):
            completions: list[str] = step["model_completions"]
            n_completions = len(completions)
            existing: list = step["scores"].get(score_key, [])

            # Trim trailing null entries so they get re-judged
            while existing and existing[-1]["value"] is None:
                existing.pop()

            if len(existing) == n_completions:
                print(f"  [behaviour {i+1}/{n}, step {j+1}] all {n_completions} completions already scored, skipping", flush=True)
                continue
            if existing:
                print(f"  [behaviour {i+1}/{n}, step {j+1}] resuming from completion {len(existing)+1}/{n_completions}", flush=True)

            step["scores"][score_key] = existing
            step_start = time.time()
            remaining = completions[len(existing):]

            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start : batch_start + batch_size]
                t0 = time.time()
                batch_criteria = evaluator.evaluate_batch(question, batch)
                elapsed = time.time() - t0
                per_item = elapsed / len(batch)

                for m, criteria in enumerate(batch_criteria):
                    k = len(existing) + batch_start + m
                    score_entry = {
                        "value":  criteria[0].value  if criteria else None,
                        "reason": criteria[0].reason if criteria else None,
                    }
                    step["scores"][score_key].append(score_entry)
                    value = score_entry["value"]
                    tag = "null" if value is None else f"score={value:.3f}"
                    print(f"  [step {j+1}, completion {k+1}/{n_completions}] {tag}  ({per_item:.1f}s avg)", flush=True)

                save(data, path)

            step_elapsed = time.time() - step_start
            scores = [s["value"] for s in step["scores"][score_key] if s["value"] is not None]
            mean_str = f"{sum(scores)/len(scores):.2f}" if scores else "n/a"
            max_str = str(max(scores)) if scores else "n/a"
            print(f"  [behaviour {i+1}/{n}, step {j+1}] done — "
                  f"{n_completions} completions in {step_elapsed:.1f}s | "
                  f"max={max_str}, mean={mean_str}", flush=True)

    return data


def main():
    parser = argparse.ArgumentParser(description="Run GuidedBench evaluation on run.json files.")
    parser.add_argument("paths", nargs="+", help="Paths to run.json files to evaluate.")
    parser.add_argument("--model", default="gemma-4-12b", help="Judge model nickname (see GuidedBench/Evaluation/utils/model.py).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of completions to judge in one forward pass.")
    args = parser.parse_args()

    ensure_db(GUIDEDBENCH_CSV, GUIDEDBENCH_DB)

    evaluator = Evaluator_GuidedBench(
        model=args.model,
        whitebox=True,
        max_new_tokens=args.max_new_tokens,
        db_path=GUIDEDBENCH_DB,
    )

    for path in args.paths:
        print(f"Evaluating {path} ...")
        data = load_run(path)
        if data is None:
            continue
        judge_run(data=data, evaluator=evaluator, score_key=f"guidedbench_{args.model}", path=path, batch_size=args.batch_size)
        print(f"Done: {path}")


if __name__ == "__main__":
    main()
