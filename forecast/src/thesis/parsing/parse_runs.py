#!/usr/bin/env python3
"""
CLI for turning raw AdversariaLLM run.json output folders into CSVs.

Subcommands
-----------
summary   Scan one or more date folders under BASE_PATH and append their
          run configs to config_summary.csv (generate_config_summary.py).
compile   Build an experiment CSV either by matching a reference run.json's
          config against config_summary.csv, or by scanning a folder of
          numbered subdirectories directly (compile_single_experiment_csv.py).
flatten   Flatten run.json files from numbered subdirectories 0-9 under a
          base path into a single per-step CSV (parse_run_jsons_to_csv.py).
scores    Load run.json scores for a compiled experiment CSV and cache them
          under data/scores/<csv_stem>/ (thesis.analysis.analyse.save_scores).
forecast  Fit per-behaviour judge-score/ASR@k forecasts from a cached
          data/scores/<run>/ and save them under data/forecasts/<run>/
          (thesis.parsing.build_forecast_summary).

Usage:
    thesis-parse summary <date_folder> [<date_folder> ...]
    thesis-parse compile <path_to_reference_run.json>
    thesis-parse compile --from-folder <folder_path> [output.csv]
    thesis-parse compile --inspect <path_to_file.parquet>
    thesis-parse flatten <base_path> [output.csv]
    thesis-parse scores <experiment_csv> [--num-behaviours N]
    thesis-parse forecast <run> --k <K> --cutoffs 50,100,200,400,800,1600
"""

import argparse
import copy
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

BASE_PATH = "/ceph/hdd/shared/llm_attacks/lapan/outputs" # change this to lapan / dornb as needed!
DATA_PATH = "/nfs/homedirs/lapan/thesis/data/experiment_overview"
SUMMARY_CSV = os.path.join(DATA_PATH, "config_summary_all_lapan.csv") # change this also as needed!
SCORES_DIR = "/nfs/homedirs/lapan/thesis/data/scores"
FORECASTS_DIR = "/nfs/homedirs/lapan/thesis/data/forecasts"

SUMMARY_FIELDNAMES = [
    "layer1_folder",
    "layer2_folder",
    "full_path",
    "model",
    "dataset",
    "attack",
    "num_steps",
    "num_return_sequences",
    "num_numbered_folders",
]

COMPILE_FIELDNAMES = [
    "layer1_folder",
    "layer2_folder",
    "full_path",
    "model",
    "dataset",
    "attack",
    "idx",
    "num_steps",
    "num_return_sequences",
    "num_numbered_folders",
]


# =============================================================================
# summary — scan date folders into config_summary.csv
# =============================================================================


def scan_date_folder(date_folder: str):
    date_path = os.path.join(BASE_PATH, date_folder)
    rows = []
    for time_entry in sorted(os.scandir(date_path), key=lambda e: e.name):
        if not time_entry.is_dir():
            continue
        time_folder = time_entry.name
        full_path = f"{date_folder}/{time_folder}"
        abs_full = os.path.join(BASE_PATH, full_path)

        numbered = [e for e in os.scandir(abs_full) if e.is_dir() and re.fullmatch(r"\d+", e.name)]
        num_numbered = len(numbered)
        if num_numbered == 0:
            continue

        # Read config from the first numbered folder (index 0)
        ref_json = os.path.join(abs_full, "0", "run.json")
        try:
            with open(ref_json) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Warning: skipping {ref_json}: {e}")
            continue

        config = data.get("config", {})
        attack_params = config.get("attack_params", {})
        num_return_sequences = str(
            attack_params.get("generation_config", {}).get("num_return_sequences", "")
        )

        rows.append(
            {
                "layer1_folder": date_folder,
                "layer2_folder": time_folder,
                "full_path": full_path,
                "model": config.get("model", ""),
                "dataset": config.get("dataset", ""),
                "attack": config.get("attack", ""),
                "num_steps": str(attack_params.get("num_steps", "")),
                "num_return_sequences": num_return_sequences,
                "num_numbered_folders": num_numbered,
            }
        )
        print(
            f"  {full_path}: {config.get('model', '?')} | {config.get('attack', '?')} | idx_count={num_numbered}"
        )

    return rows


def cmd_summary(args: argparse.Namespace) -> None:
    all_rows = []
    for date_folder in args.date_folders:
        print(f"\nScanning {date_folder}...")
        all_rows.extend(scan_date_folder(date_folder))

    # Append to existing CSV if it exists, otherwise create fresh
    existing = []
    existing_keys = set()
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
                existing_keys.add(row["full_path"])

    new_rows = [r for r in all_rows if r["full_path"] not in existing_keys]
    all_combined = existing + new_rows

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_combined)

    print(f"\nWrote {len(all_combined)} total rows ({len(new_rows)} new) to {SUMMARY_CSV}")


# =============================================================================
# compile — match a reference run.json's config against config_summary.csv,
# or scan a folder of numbered subdirectories directly
# =============================================================================


def inspect_parquet(parquet_path: str):
    """
    Print column names and a row count breakdown by model-attack combination
    for a given parquet file.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow is required. Install it with: pip install pyarrow")
        sys.exit(1)

    print(f"Inspecting: {parquet_path}\n")

    schema = pq.read_schema(parquet_path)
    print("Columns:")
    for i, name in enumerate(schema.names):
        print(f"  {i + 1:>3}. {name} ({schema.field(name).type})")

    table = pq.read_table(parquet_path, columns=["model", "attack", "num_steps", "num_streams"])
    print(f"\nTotal rows: {table.num_rows:,}\n")

    import collections

    model_col = table.column("model").to_pylist()
    attack_col = table.column("attack").to_pylist()
    steps_col = table.column("num_steps").to_pylist()
    streams_col = table.column("num_streams").to_pylist()

    rows_count = collections.Counter(zip(model_col, attack_col))

    steps_map = collections.defaultdict(set)
    streams_map = collections.defaultdict(set)
    for model, attack, steps, streams in zip(model_col, attack_col, steps_col, streams_col):
        key = (model, attack)
        if steps is not None:
            steps_map[key].add(int(steps))
        if streams is not None:
            streams_map[key].add(int(streams))

    def _fmt_set(s):
        if not s:
            return "?"
        if len(s) == 1:
            return str(next(iter(s)))
        return f"{min(s)}–{max(s)}"

    print(f"{'Model':<40} {'Attack':<30} {'Rows':>6}  {'Steps':>6}  {'Samples':>8}")
    print("-" * 96)
    for (model, attack), count in sorted(rows_count.items()):
        key = (model, attack)
        print(
            f"{str(model):<40} {str(attack):<30} {count:>6,}  {_fmt_set(steps_map[key]):>6}  {_fmt_set(streams_map[key]):>8}"
        )


def load_reference_config(json_path: str) -> dict[str, Any]:
    try:
        with open(json_path) as f:
            data = json.load(f)
        config = data.get("config")
        if config is None:
            print(f"Error: No 'config' key found in {json_path}")
            sys.exit(1)
        return config
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON from {json_path}: {e}")
        sys.exit(1)


def configs_match(ref_config: dict[str, Any], candidate_config: dict[str, Any]) -> bool:
    """
    Return True if the candidate config matches the reference config on all keys,
    except config["dataset_params"]["idx"] which is intentionally ignored.
    """
    # if candidate_config["attack_params"]["loss"] == "ce":
    #     return True
    for key, ref_val in ref_config.items():
        cand_val = candidate_config.get(key)
        if key == "dataset_params":
            ref_sans_idx = {k: v for k, v in ref_val.items() if k != "idx"}
            cand_sans_idx = {k: v for k, v in (cand_val or {}).items() if k != "idx"}
            if ref_sans_idx != cand_sans_idx:
                return False
        else:
            if cand_val != ref_val:
                return False
    return True


def get_idx(config: dict[str, Any]) -> str:
    return str(config.get("dataset_params", {}).get("idx", ""))


def load_candidates_from_csv(ref_config: dict[str, Any]) -> list[dict[str, str]]:
    """
    Read config_summary.csv and return rows whose coarse fields
    (model, dataset, attack, num_steps, num_return_sequences) match the reference.
    """
    ref_model = ref_config.get("model", "")
    ref_dataset = ref_config.get("dataset", "")
    ref_attack = ref_config.get("attack", "")
    ref_num_steps = str(ref_config.get("attack_params", {}).get("num_steps", ""))
    ref_num_return_sequences = str(
        ref_config.get("attack_params", {})
        .get("generation_config", {})
        .get("num_return_sequences", "")
    )

    candidates = []
    try:
        with open(SUMMARY_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row["model"] == ref_model
                    and row["dataset"] == ref_dataset
                    and row["attack"] == ref_attack
                    and str(row["num_steps"]) == ref_num_steps
                    and str(row["num_return_sequences"]) == ref_num_return_sequences
                ):
                    candidates.append(row)
    except FileNotFoundError:
        print(f"Error: Summary CSV not found at {SUMMARY_CSV}")
        sys.exit(1)

    return candidates


def check_candidate(row: dict[str, str], ref_config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Do a full config comparison on a candidate row's run.json.
    Returns a result dict (with idx) if it matches, else None.
    """
    run_json_path = os.path.join(BASE_PATH, row["full_path"], "0", "run.json")

    try:
        with open(run_json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not read {run_json_path}: {e}")
        return None

    candidate_config = copy.deepcopy(data.get("config", {}))
    candidate_config["dataset_params"]["idx"] = ref_config["dataset_params"]["idx"]
    candidate_config["attack_params"]["seed"] = ref_config["attack_params"]["seed"]

    # resume_from points at a prior checkpoint folder that's specific to each
    # behavior/run, just like idx and seed — compare only whether it's set,
    # not its specific path, so resumed parallel runs still match on
    # everything else while a resumed run doesn't match a fresh one.
    candidate_config["attack_params"]["resume_from"] = bool(
        candidate_config["attack_params"].get("resume_from")
    )
    ref_config_for_compare = copy.deepcopy(ref_config)
    ref_config_for_compare["attack_params"]["resume_from"] = bool(
        ref_config_for_compare["attack_params"].get("resume_from")
    )

    if not configs_match(ref_config_for_compare, candidate_config):
        return None

    return {
        "layer1_folder": row["layer1_folder"],
        "layer2_folder": row["layer2_folder"],
        "full_path": row["full_path"],
        "model": row["model"],
        "dataset": row["dataset"],
        "attack": row["attack"],
        "idx": get_idx(candidate_config),
        "num_steps": row["num_steps"],
        "num_return_sequences": row["num_return_sequences"],
        "num_numbered_folders": row["num_numbered_folders"],
    }


def save_to_csv(results: list[dict[str, Any]], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPILE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} record(s) to {output_path}")


def compile_from_folder(folder_path: str, output_path: str | None = None) -> list[dict[str, Any]]:
    """
    Build an experiment CSV from a folder whose immediate children are numbered
    subdirectories (0/, 1/, 2/, ...), each containing a run.json.

    This is the layout produced by claudini-style runs where every numbered
    folder is one experiment index. Call example:

        thesis-parse compile --from-folder AdversariaLLM/outputs/2026-06-11/19-11-11/
    """
    folder_path = os.path.abspath(folder_path)

    parts = folder_path.rstrip(os.sep).split(os.sep)
    layer2_folder = parts[-1] if len(parts) >= 1 else ""
    layer1_folder = parts[-2] if len(parts) >= 2 else ""

    numbered_dirs: list[tuple] = []
    for entry in os.scandir(folder_path):
        if entry.is_dir() and re.fullmatch(r"\d+", entry.name):
            numbered_dirs.append((int(entry.name), entry.path))
    numbered_dirs.sort()

    if not numbered_dirs:
        print(f"No numbered subdirectories found in {folder_path}")
        return []

    num_numbered_folders = len(numbered_dirs)
    results = []

    for _, dir_path in numbered_dirs:
        run_json_path = os.path.join(dir_path, "run.json")
        try:
            with open(run_json_path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Warning: Could not read {run_json_path}: {e}")
            continue

        config = data.get("config", {})
        attack_params = config.get("attack_params", {})
        num_return_sequences = str(
            attack_params.get("generation_config", {}).get("num_return_sequences", "")
        )

        results.append(
            {
                "layer1_folder": layer1_folder,
                "layer2_folder": layer2_folder,
                "full_path": folder_path,
                "model": config.get("model", ""),
                "dataset": config.get("dataset", ""),
                "attack": config.get("attack", ""),
                "idx": get_idx(config),
                "num_steps": str(attack_params.get("num_steps", "")),
                "num_return_sequences": num_return_sequences,
                "num_numbered_folders": num_numbered_folders,
            }
        )

    if output_path is None:
        output_path = os.path.join(folder_path, "experiment.csv")

    save_to_csv(results, output_path)
    return results


def cmd_compile(args: argparse.Namespace) -> None:
    if args.inspect:
        inspect_parquet(args.inspect)
        return

    if args.from_folder:
        compile_from_folder(args.from_folder, args.output)
        return

    if args.ref_json_path is None:
        print(
            "Error: provide <path_to_reference_run.json>, --from-folder <dir>, or --inspect <file.parquet>"
        )
        sys.exit(1)

    ref_json_path = args.ref_json_path
    print(f"Reference file : {ref_json_path}")

    ref_config = load_reference_config(ref_json_path)
    print(f"Matching config: {json.dumps(ref_config, indent=2)}\n")

    print(f"Pre-filtering candidates from {SUMMARY_CSV}...")
    candidates = load_candidates_from_csv(ref_config)
    print(f"Found {len(candidates)} coarse-match candidate(s). Running full config check...\n")

    results = []
    for row in candidates:
        result = check_candidate(row, ref_config)
        if result is not None:
            results.append(result)

    if not results:
        print("No matching runs found.")
        return

    # ref_dir = os.path.dirname(os.path.abspath(ref_json_path))
    output_path = os.path.join(DATA_PATH, "matching_runs.csv")
    save_to_csv(results, output_path)

    print("\nSample records:")
    for i, r in enumerate(results[:3]):
        print(
            f"  {i + 1}. {r['full_path']}: {r['model']} | {r['dataset']} | "
            f"{r['attack']} | idx={r['idx']} | {r['num_steps']} steps | "
            f"{r['num_return_sequences']} seqs | {r['num_numbered_folders']} folders"
        )
    if len(results) > 3:
        print(f"  ... and {len(results) - 3} more record(s)")


# =============================================================================
# scores — load run.json scores for a compiled experiment CSV and cache them
# =============================================================================


def cmd_scores(args: argparse.Namespace) -> None:
    # Imported lazily so the other subcommands stay free of numpy/matplotlib/tqdm.
    from thesis.analysis.analyse import save_scores

    save_scores(
        args.csv,
        scores_dir=args.scores_dir,
        base_path=args.base_path,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        num_behaviours=args.num_behaviours,
        filter_seed=args.filter_seed,
        force=args.force,
    )


# =============================================================================
# forecast — fit per-behaviour judge-score/ASR@k forecasts from cached scores
# and save them under data/forecasts/<run>/ (build_forecast_summary)
# =============================================================================


def cmd_forecast(args: argparse.Namespace) -> None:
    # Imported lazily so the other subcommands stay free of numpy/scipy/joblib.
    from thesis.parsing.build_forecast_summary import build_forecast_outputs

    families = args.families.split(",") if args.families else None
    cutoffs = [int(c) for c in args.cutoffs.split(",")]

    build_forecast_outputs(
        args.run,
        scores_dir=args.scores_dir,
        out_dir=args.out_dir,
        k=args.k,
        cutoffs=cutoffs,
        threshold=args.threshold,
        families=families,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


# =============================================================================
# flatten — run.json files under numbered subdirectories 0-9 -> one CSV
# =============================================================================


def parse_run_jsons_to_csv(base_path: str | Path, output_path: str | Path = "output.csv") -> None:
    """
    Reads run.json files from subdirectories named 0-9 under base_path,
    flattens the relevant fields, and writes them to a single CSV file.

    Args:
        base_path:   Path to the directory containing subdirectories 0-9.
        output_path: Path for the output CSV file (default: output.csv).
    """
    base_path = Path(base_path)
    output_path = Path(output_path)

    expected_dirs = [str(i) for i in range(10)]
    rows = []

    for folder_name in expected_dirs:
        folder = base_path / folder_name
        json_file = folder / "run.json"

        if not folder.exists():
            warnings.warn(f"Expected subdirectory '{folder}' does not exist — skipping.")
            continue
        if not json_file.exists():
            warnings.warn(f"'run.json' not found in '{folder}' — skipping.")
            continue

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        original_prompt = data["runs"][0]["original_prompt"][0]["content"]

        steps = data["runs"][0]["steps"]

        if len(steps) != 250:
            warnings.warn(f"[{folder_name}] Expected 250 steps, got {len(steps)}.")

        for item in steps:
            step = item["step"]

            model_input = item["model_input"][0]["content"]

            model_completions = item["model_completions"]
            if len(model_completions) > 1:
                warnings.warn(
                    f"[{folder_name}] Step {step}: 'model_completions' has "
                    f"{len(model_completions)} entries (expected 1). "
                    f"Using the first entry."
                )
            model_completion = model_completions[0] if model_completions else None

            scores = item["scores"]
            strong_reject_score = scores.get("strong_reject").get("p_harmful")[0]
            harmbench_score = scores.get("harmbench").get("p_harmful")[0]

            rows.append(
                {
                    "run_id": folder_name,
                    "step": step,
                    "behavior_for_judge": original_prompt,
                    "target_model_input_text": model_input,
                    "model_target_model_output_text": model_completion,
                    "judge_score_strong_reject": strong_reject_score,
                    "judge_score_harmbench": harmbench_score,
                }
            )

    if not rows:
        warnings.warn("No data was collected — CSV file will not be written.")
        return

    fieldnames = [
        "run_id",
        "step",
        "behavior_for_judge",
        "target_model_input_text",
        "model_target_model_output_text",
        "judge_score_strong_reject",
        "judge_score_harmbench",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to '{output_path}'.")


def cmd_flatten(args: argparse.Namespace) -> None:
    parse_run_jsons_to_csv(args.base_path, args.output)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Parse AdversariaLLM run.json output folders into CSVs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="Scan date folders into config_summary.csv")
    p_summary.add_argument(
        "date_folders", nargs="+", help="Date folder(s) under BASE_PATH, e.g. 2026-06-15"
    )
    p_summary.set_defaults(func=cmd_summary)

    p_compile = sub.add_parser("compile", help="Build an experiment CSV")
    p_compile.add_argument(
        "ref_json_path",
        nargs="?",
        help="Path to a reference run.json to match against config_summary.csv",
    )
    p_compile.add_argument(
        "--from-folder", help="Scan a folder of numbered subdirectories directly"
    )
    p_compile.add_argument("--inspect", help="Print column/row summary of a parquet file")
    p_compile.add_argument(
        "output", nargs="?", default=None, help="Output CSV path (used with --from-folder)"
    )
    p_compile.set_defaults(func=cmd_compile)

    p_flatten = sub.add_parser(
        "flatten", help="Flatten numbered run.json subdirectories into one CSV"
    )
    p_flatten.add_argument("base_path", help="Directory containing subdirectories 0-9")
    p_flatten.add_argument("output", nargs="?", default="output.csv", help="Output CSV path")
    p_flatten.set_defaults(func=cmd_flatten)

    p_scores = sub.add_parser(
        "scores", help="Load run.json scores for a compiled experiment CSV and cache them under data/scores/"
    )
    p_scores.add_argument("csv", help="Path to a compiled experiment CSV")
    p_scores.add_argument("--base-path", default=BASE_PATH, help="Base path run.json folders are resolved relative to")
    p_scores.add_argument("--scores-dir", default=SCORES_DIR)
    p_scores.add_argument(
        "--num-behaviours",
        type=int,
        default=None,
        help="Expected behaviours per run. The CSV's own num_numbered_folders value is "
        "often wrong, so pass this explicitly rather than relying on the default.",
    )
    p_scores.add_argument(
        "--num-steps", type=int, default=None, help="Number of steps to allocate (default: inferred from run data)"
    )
    p_scores.add_argument("--num-samples", type=int, default=32, help="Samples per step, hard-capped at 50 in run.json")
    p_scores.add_argument("--filter-seed", type=int, default=None)
    p_scores.add_argument("--force", action="store_true", help="Recompute even if a valid cache already exists")
    p_scores.set_defaults(func=cmd_scores)

    p_forecast = sub.add_parser(
        "forecast",
        help="Fit per-behaviour judge-score/ASR@k forecasts from cached scores and save them "
        "under data/forecasts/<run>/",
    )
    p_forecast.add_argument("run", help="Run name -- the <run> in data/scores/<run>/all_scores.npy")
    p_forecast.add_argument("--scores-dir", default=SCORES_DIR)
    p_forecast.add_argument("--out-dir", default=FORECASTS_DIR)
    p_forecast.add_argument("--k", type=int, required=True, help="Sample count for the ASR@k trajectory")
    p_forecast.add_argument(
        "--cutoffs",
        required=True,
        help="Comma-separated numeric holdout cutoffs, e.g. 50,100,200,400,800,1600 "
        "(\"max\" is always added implicitly)",
    )
    p_forecast.add_argument("--threshold", type=float, default=0.5, help="Judge-score success threshold")
    p_forecast.add_argument(
        "--families",
        default=None,
        help="Comma-separated family names (default: every family in forecast.ALL_FAMILY_NAMES)",
    )
    p_forecast.add_argument("--n-bootstrap", type=int, default=200, help="Bootstrap replicates for ASR@k")
    p_forecast.add_argument("--seed", type=int, default=0)
    p_forecast.add_argument("--n-jobs", type=int, default=-1, help="Passed to joblib.Parallel (-1 = all cores)")
    p_forecast.set_defaults(func=cmd_forecast)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
