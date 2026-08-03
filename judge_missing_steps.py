#!/usr/bin/env python3
"""
Judge whichever steps in the given run.json file(s) are missing scores for a
given classifier, e.g. steps added by a resume/continuation whose
completions were generated but never scored.

Unlike run_judges.py (which decides "already scored" by checking only
subrun["steps"][0]["scores"], and judges every step in the subrun as one
batch), this checks each step individually and judges only the completions
that step is missing — so already-scored steps are left untouched and
partially-scored steps are topped up rather than re-judged from scratch.

Usage:
    python judge_missing_steps.py path/to/run.json [path/to/another/run.json ...]
    python judge_missing_steps.py --classifier strong_reject path/to/run.json
"""

import argparse
import copy
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True

from judgezoo import Judge


def judge_missing_steps(path: str, classifier: str, batch_size: int = 32) -> None:
    with open(path) as f:
        attack_run = json.load(f)

    judge = None
    for subrun in attack_run["runs"]:
        original_conversation = subrun["original_prompt"]

        # (step, completion) pairs still missing a score for this classifier
        todo = []
        for step in subrun["steps"]:
            completions = step["model_completions"]
            existing = step["scores"].get(classifier, {}).get("p_harmful", [])
            if len(existing) >= len(completions):
                continue
            for completion in completions[len(existing):]:
                todo.append((step, completion))

        if not todo:
            continue

        if judge is None:
            print("Loading judge...")
            judge = Judge.from_name(classifier)

        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start : batch_start + batch_size]
            modified_prompts = []
            for step, completion in batch:
                model_input = step["model_input"]
                modified_prompt = copy.deepcopy(original_conversation)
                if modified_prompt[-1]["role"] == "assistant":
                    modified_prompt[-1]["content"] = model_input[-1]["content"] + completion
                else:
                    modified_prompt.append({"role": "assistant", "content": completion})
                modified_prompts.append(modified_prompt)

            results = judge(modified_prompts)

            for i, (step, _) in enumerate(batch):
                step_scores = step["scores"].setdefault(classifier, {})
                for k, v in results.items():
                    step_scores.setdefault(k, []).append(v[i])

            with open(path, "w") as f:
                json.dump(attack_run, f, indent=2)

            print(f"  {path}: judged {batch_start + len(batch)}/{len(todo)} missing completions")

    print(f"Done: {path}")


def main():
    parser = argparse.ArgumentParser(description="Judge steps missing scores for a classifier.")
    parser.add_argument("paths", nargs="+", help="Paths to run.json files.")
    parser.add_argument("--classifier", default="strong_reject")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    for path in args.paths:
        judge_missing_steps(path, args.classifier, args.batch_size)


if __name__ == "__main__":
    main()
