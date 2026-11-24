import csv
import json
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

BASE_PATH = "/ceph/hdd/shared/llm_attacks"
OUTPUTS_BASE_PATH = "/ceph/hdd/shared/llm_attacks/dornb/outputs"
SCORES_DIR = "/nfs/homedirs/lapan/thesis/data/scores"
SCORE_KEY = "strong_reject"
METRIC_KEY = "p_harmful"
HARMBENCH_CSV = (
    "/nfs/homedirs/lapan/AdversariaLLM/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
CATEGORY_LOOKUP_PATH = "/nfs/homedirs/lapan/thesis/data/harmbench_categories_lookup.csv"

# Computer Modern (LaTeX's own default typeface) is bundled with matplotlib, so this needs no
# external font install. mathtext.fontset="cm" keeps math elements (e.g. R^2) in the same
# family; axes.unicode_minus is off since cmr10 lacks the Unicode minus glyph.
_LATEX_FONT_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
}

# Text scale for figures meant to stand on their own in the printed thesis. Legend is kept a
# notch below title/axis text (so it doesn't dominate the panel) but never below the 18pt floor.
_BIG_FONT = 20
_BIG_TITLE_FONT = 24
_BIG_LEGEND_FONT = 18

# Cap for per-line/per-category viridis sampling -- see the matching constant/comment in
# plot_forecasts.py. Duplicated rather than imported for the same reason as the font constants
# above: no cross-import between this file and plot_forecasts.py.
_VIRIDIS_LINE_MAX = 0.85

# Soft dark-gray (rather than pure black) edges/ticks, matching `_SLEEK_RC` in plot_forecasts.py
# -- kept as a separate dict (not merged into `_LATEX_FONT_RC`) since not every plot in this
# file has opted into the shared house style yet.
_SLEEK_EXTRA_RC = {
    "axes.linewidth": 1.3,
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
}


def _save_fig(fig, title, default_name, save=True):
    """Save `fig` to `plots/<slug>.pdf` when `save` -- `<slug>` is `title` sanitized to a safe
    filename if given, else `default_name`. Creates the `plots/` directory if it doesn't exist.
    Duplicated from plot_forecasts.py's identical helper (no cross-import between the two files).
    """
    if not save:
        return
    os.makedirs("plots", exist_ok=True)
    slug = re.sub(r"[^\w\-]+", "_", title).strip("_") if title else default_name
    fig.savefig(f"plots/{slug}.pdf", bbox_inches="tight")


def _is_parquet(path: str) -> bool:
    return path.endswith(".parquet")


def load_scores(csv_path, num_steps, filters=None, behaviour_ids=None):
    """Load score data from either a CSV (pointing to run.json files) or a parquet file.
    Returns a list of (num_steps, 50) arrays, one per behaviour.

    filters: only used for parquet. A dict of column->value pairs to filter rows, e.g.:
        {"model": "meta-llama/Llama-2-7b-chat-hf", "attack": "gcg"}
    behaviour_ids: optional list of dataset idx values to restrict which behaviours are loaded (CSV only).
    """
    if _is_parquet(csv_path):
        return _load_scores_parquet(csv_path, num_steps, filters=filters)
    if filters:
        print("Warning: filters are only supported for parquet files and will be ignored.")
    return _load_scores_csv(csv_path, num_steps, behaviour_ids=behaviour_ids)


def _load_scores_parquet(parquet_path, num_steps, filters=None):
    """Load scores from a parquet file.
    Each row is treated as one behaviour. The column 'judge_scores_by_step_json'
    contains a list (indexed by step) of {SCORE_KEY: {METRIC_KEY: [floats]}} dicts.

    filters: optional dict of column->value pairs, e.g. {"model": "...", "attack": "..."}
    """
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for parquet support. Install with: pip install pyarrow"
        )

    print(f"Loading parquet: {parquet_path}")

    # Build pyarrow filters from the dict, e.g. [("model", "=", "..."), ("attack", "=", "...")]
    arrow_filters = None
    if filters:
        arrow_filters = [(col, "=", val) for col, val in filters.items()]
        print(f"Applying filters: {filters}")

    cols = (
        list(filters.keys()) + ["judge_scores_by_step_json"]
        if filters
        else ["judge_scores_by_step_json"]
    )
    table = pq.read_table(parquet_path, columns=cols, filters=arrow_filters)
    col = table.column("judge_scores_by_step_json").to_pylist()
    print(f"Loaded {len(col)} row(s) from parquet.")

    all_scores = []
    skipped = 0
    for raw in tqdm(col, desc="Parsing parquet rows"):
        scores = _parse_judge_scores_json(raw, num_steps)
        if scores is not None:
            all_scores.append(scores)
        else:
            skipped += 1

    print(f"Successfully parsed {len(all_scores)} / {len(col)} row(s). ({skipped} skipped)")
    return all_scores


def _parse_judge_scores_json(raw, num_steps, num_samples):
    """Parse one value of judge_scores_by_step_json into a (num_steps, 50) array.
    The value is a list of per-step dicts: [{SCORE_KEY: {METRIC_KEY: [floats]}}, ...]
    Step index = position in the list.
    """
    if raw is None:
        return None

    # May be a string (if stored as JSON text) or already a Python list
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Warning: could not parse judge_scores_by_step_json: {e}")
            return None

    if not isinstance(raw, list):
        print(f"  Warning: expected list, got {type(raw)}")
        return None

    scores = np.full((num_steps, num_samples), np.nan)
    for idx, step_data in enumerate(raw):
        if idx >= num_steps:
            break
        try:
            p = step_data[SCORE_KEY][METRIC_KEY]
            n = min(len(p), num_samples)
            scores[idx, :n] = p[:n]
        except (KeyError, TypeError):
            pass

    return scores


def _load_scores_csv(csv_path, num_steps, behaviour_ids=None):
    """Original CSV-based loader. Returns a list of (num_steps, 50) arrays."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} row(s) from {csv_path}")

    all_scores = []
    for row in tqdm(rows, desc="Loading run.json files"):
        full_path = row["full_path"].strip()
        if full_path.endswith(".json"):
            paths = [os.path.join(BASE_PATH, full_path)]
        else:
            num_folders = int(row.get("num_numbered_folders", 1) or 1)
            paths = [
                os.path.join(BASE_PATH, full_path, str(i), "run.json") for i in range(num_folders)
            ]
        for path in paths:
            scores = _load_run(path, num_steps, behaviour_ids=behaviour_ids)
            if scores is not None:
                all_scores.append(scores)

    print(f"Successfully loaded {len(all_scores)} / {len(rows)} file(s).")
    return all_scores


def _load_run(run_json_path, num_steps, num_samples, behaviour_ids=None):
    """Read one run.json; return (num_steps, 50) array or None on failure."""
    try:
        with open(run_json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: {run_json_path}: {e}")
        return None

    if behaviour_ids is not None:
        try:
            raw_idx = data["config"]["dataset_params"]["idx"]
            idx = raw_idx[0] if isinstance(raw_idx, list) else raw_idx
            if int(idx) not in behaviour_ids:
                return None
        except (KeyError, TypeError, ValueError, IndexError):
            pass  # can't determine idx; include the run

    runs = data.get("runs", [])
    if not runs:
        print(f"  Warning: no 'runs' key in {run_json_path}")
        return None

    step_list = runs[0].get("steps", [])
    scores = np.full((num_steps, num_samples), np.nan)
    for step_data in step_list:
        idx = step_data.get("step")
        if idx is None or idx >= num_steps:
            continue
        try:
            p = step_data["scores"][SCORE_KEY][METRIC_KEY]
            n = min(len(p), num_samples)
            scores[idx, :n] = p[:n]
        except (KeyError, TypeError):
            pass
    return scores


def build_category_lookup(
    folder="/ceph/hdd/shared/llm_attacks/dataset/reinforce_llama-3-1",
    harmbench_csv_path=HARMBENCH_CSV,
    output_path=CATEGORY_LOOKUP_PATH,
):
    """Build harmbench_categories_lookup.csv from a folder of per-behaviour run.json files.

    For each .json in folder, reads config.dataset_params.idx and
    runs[0].original_prompt[0].content, matches the prompt to the Behavior column in
    harmbench_csv_path, and writes idx / prompt / FunctionalCategory / SemanticCategory.
    """
    harmbench = []
    with open(harmbench_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            harmbench.append(
                (
                    row["Behavior"],
                    {
                        "FunctionalCategory": row["FunctionalCategory"],
                        "SemanticCategory": row["SemanticCategory"],
                    },
                )
            )

    def _find_meta(prompt):
        for behavior, meta in harmbench:
            if prompt.endswith(behavior):
                return meta
        return {"FunctionalCategory": "", "SemanticCategory": ""}

    rows_out = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue
        file_path = os.path.join(folder, fname)
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: {fname}: {e}")
            continue

        idx_raw = data["config"]["dataset_params"]["idx"]
        idx = idx_raw[0] if isinstance(idx_raw, list) else idx_raw
        prompt = data["runs"][0]["original_prompt"][0]["content"]
        meta = _find_meta(prompt)
        if not meta["FunctionalCategory"]:
            print(f"Warning: no harmbench match for {fname}: {prompt[:60]!r}")

        rel_path = os.path.relpath(file_path, BASE_PATH)
        rows_out.append(
            {
                "full_path": rel_path,
                "idx": idx,
                "prompt": prompt,
                "FunctionalCategory": meta["FunctionalCategory"],
                "SemanticCategory": meta["SemanticCategory"],
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["full_path", "idx", "prompt", "FunctionalCategory", "SemanticCategory"]
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {output_path}")
    return rows_out


def load_categories_for_csv(csv_path, lookup_path=CATEGORY_LOOKUP_PATH):
    """Return a list of SemanticCategory strings, one per score array loaded by _load_scores_csv.

    The order matches the order returned by _load_scores_csv / _load_scores_csv_with_flops.
    Matching is done by behavior idx (handles both plain integers and "[0]"-style list strings).
    Rows with no match get SemanticCategory="unknown".
    """
    idx_to_cat = {}
    with open(lookup_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                idx_to_cat[int(row["idx"])] = row["SemanticCategory"]
            except (ValueError, KeyError):
                pass

    def _parse_idx(raw):
        raw = (raw or "").strip()
        if raw.startswith("["):
            import ast

            val = ast.literal_eval(raw)
            return val[0] if isinstance(val, list) and val else None
        try:
            return int(raw)
        except ValueError:
            return None

    categories = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        full_path = row["full_path"].strip()
        idx = _parse_idx(row.get("idx", ""))
        cat = idx_to_cat.get(idx, "unknown")

        if full_path.endswith(".json"):
            categories.append(cat)
        else:
            num_folders = int(row.get("num_numbered_folders", 1) or 1)
            categories.extend([cat] * num_folders)

    return categories


def _nanmaximum_accumulate(arr, axis):
    """Like np.maximum.accumulate but treats NaN as missing rather than propagating it."""
    result = arr.copy()
    arr_m = np.moveaxis(arr, axis, -1)
    res_m = np.moveaxis(result, axis, -1)
    running = np.full(arr_m.shape[:-1], -np.inf)
    for i in range(arr_m.shape[-1]):
        valid = ~np.isnan(arr_m[..., i])
        running = np.where(valid, np.maximum(running, arr_m[..., i]), running)
        res_m[..., i] = np.where(running == -np.inf, np.nan, running)
    return result


def compute_metrics(all_scores):
    """Return (mean_avg, mean_max, mean_1st, mean_cummax_1st, mean_var), each shape (num_steps,)."""
    stacked = np.stack(all_scores, axis=0)  # (n_files, num_steps, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_avg = np.nanmean(np.nanmean(stacked, axis=2), axis=0)
        mean_max = np.nanmean(np.nanmax(stacked, axis=2), axis=0)
        mean_1st = np.nanmean(stacked[:, :, 0], axis=0)
        cummax_1st = _nanmaximum_accumulate(stacked[:, :, 0], axis=1)
        mean_cummax_1st = np.nanmean(cummax_1st, axis=0)
        mean_var = np.nanmean(np.nanvar(stacked, axis=2), axis=0)
    return mean_avg, mean_max, mean_1st, mean_cummax_1st, mean_var


def _read_attack_seed(run_json_path):
    """Return the attack seed from a run.json, or None if unreadable."""
    try:
        with open(run_json_path) as f:
            return json.load(f)["config"]["attack_params"]["seed"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _read_seed_and_idx(run_json_path):
    """Return (attack_seed, behaviour_idx) from a run.json, or (None, None) if unreadable."""
    try:
        with open(run_json_path) as f:
            data = json.load(f)
        seed = data["config"]["attack_params"]["seed"]
        raw_idx = data["config"]["dataset_params"]["idx"]
        idx = raw_idx[0] if isinstance(raw_idx, list) else raw_idx
        return seed, int(idx)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
        return None, None


def _build_virtual_runs(rows, n_beh, base_path):
    """Map CSV rows to virtual runs, each a list of length n_beh whose entries
    are either (full_path, subfolder_idx) pairs or None, ordered by behaviour
    rank 0, 1, 2, ... Also returns a parallel list of each virtual run's attack
    seed (None if it couldn't be determined).

    Handles three on-disk layouts:

      * a folder with exactly n_beh numbered subfolders (one sequential job
        that ran the whole behaviour list) — subfolder order is behaviour rank.
      * a folder with more than n_beh numbered subfolders, from several seeds
        landing in the same timestamped folder — split by attack seed, sorted
        subfolder order within a seed gives behaviour rank.
      * folders with fewer than n_beh numbered subfolders — the layout produced
        by per-behaviour Hydra sweeps (job-array submission), where each task
        gets its own timestamped folder (occasionally a handful of tasks that
        started in the same second share one folder). These are grouped
        *across* folders by attack seed, and ranked by actual dataset idx
        (ascending, over the union of idx seen across all seeds) so a given
        behaviour lands in the same slot for every seed. A seed missing some
        idx (e.g. a failed job) gets None in those slots instead of raising;
        a warning is printed listing how many seeds are actually present for
        each affected idx.
    """
    virtual_runs = []
    run_seeds = []
    loose_entries = []  # (full_path, subfolder_idx) from folders with < n_beh subfolders

    for row in rows:
        full_path = row["full_path"].strip()
        total_folders = int(row.get("num_numbered_folders", 1))

        if total_folders < n_beh:
            loose_entries.extend((full_path, i) for i in range(total_folders))
        elif total_folders == n_beh:
            virtual_runs.append([(full_path, i) for i in range(n_beh)])
            run_seeds.append(_read_attack_seed(os.path.join(base_path, full_path, "0", "run.json")))
        else:
            seed_to_indices = {}
            for subfolder_idx in range(total_folders):
                path = os.path.join(base_path, full_path, str(subfolder_idx), "run.json")
                seed = _read_attack_seed(path)
                if seed is None:
                    raise ValueError(f"Could not read attack seed from {path}")
                seed_to_indices.setdefault(seed, []).append(subfolder_idx)
            for seed, indices in seed_to_indices.items():
                if len(indices) != n_beh:
                    raise ValueError(
                        f"{full_path}: seed={seed} has {len(indices)} subfolders, expected {n_beh}."
                    )
                virtual_runs.append([(full_path, i) for i in sorted(indices)])
                run_seeds.append(seed)

    if loose_entries:
        seed_to_entries = {}
        for full_path, subfolder_idx in loose_entries:
            path = os.path.join(base_path, full_path, str(subfolder_idx), "run.json")
            seed, idx = _read_seed_and_idx(path)
            if seed is None:
                raise ValueError(f"Could not read attack seed/idx from {path}")
            seed_to_entries.setdefault(seed, []).append((idx, full_path, subfolder_idx))

        all_idxs = sorted({idx for entries in seed_to_entries.values() for idx, _, _ in entries})
        if len(all_idxs) != n_beh:
            raise ValueError(
                f"Found {len(all_idxs)} distinct behaviour idx across all seeds, "
                f"expected {n_beh}. idxs={all_idxs}"
            )
        idx_to_rank = {idx: rank for rank, idx in enumerate(all_idxs)}

        n_seeds = len(seed_to_entries)
        missing_seeds_per_idx = {idx: 0 for idx in all_idxs}
        for seed, entries in seed_to_entries.items():
            idxs_seen = [e[0] for e in entries]
            if len(idxs_seen) != len(set(idxs_seen)):
                print(
                    f"Warning: seed={seed}: duplicate idx found: {sorted(idxs_seen)}. Removing duplicate entries..."
                )
                seen_idxs = set()
                deduped_entries = []
                for entry in entries:
                    if entry[0] not in seen_idxs:
                        seen_idxs.add(entry[0])
                        deduped_entries.append(entry)
                entries = deduped_entries
                idxs_seen = [e[0] for e in entries]

            run_entries = [None] * n_beh
            for idx, fp, sub in entries:
                run_entries[idx_to_rank[idx]] = (fp, sub)
            virtual_runs.append(run_entries)
            run_seeds.append(seed)

            for idx in set(all_idxs) - set(idxs_seen):
                missing_seeds_per_idx[idx] += 1

        incomplete = {
            idx: n_seeds - missing for idx, missing in missing_seeds_per_idx.items() if missing > 0
        }
        if incomplete:
            print(
                f"Warning: {len(incomplete)}/{n_beh} behaviour(s) have data from fewer "
                f"than {n_seeds} seeds (missing runs left as NaN):"
            )
            for idx in sorted(incomplete):
                print(f"  idx={idx}: {incomplete[idx]}/{n_seeds} seeds present")

    return virtual_runs, run_seeds


def _infer_num_steps(virtual_runs, base_path):
    """Infer num_steps from the actual run data instead of a config value.

    Reads just one representative run.json -- the first non-None entry found
    in virtual_runs -- and takes the "step" field of the *last* entry in its
    runs[0]["steps"] list. This is more reliable than the CSV's num_steps
    column (the original attack config), which undercounts a run that was
    later resumed/continued past its originally configured step budget.
    Every file in one experiment is assumed to have progressed the same
    number of steps (continuation is applied experiment-wide), so one file
    suffices.
    """
    for run_entries in virtual_runs:
        for entry in run_entries:
            if entry is None:
                continue
            full_path, subfolder_idx = entry
            path = os.path.join(base_path, full_path, str(subfolder_idx), "run.json")
            try:
                with open(path) as f:
                    data = json.load(f)
                steps = data["runs"][0]["steps"]
                return int(steps[-1]["step"]) + 1
            except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
    raise ValueError("Could not infer num_steps: no readable run.json found in virtual_runs.")


def load_experiment(
    csv_path,
    num_samples,
    num_steps=None,
    num_behaviours=None,
    base_path=OUTPUTS_BASE_PATH,
    filter_seed=None,
):
    """Load all run.json files into a 4-D score array.

    This is the primary loading function. All plotting helpers accept its output.

    Axis layout
    -----------
    scores : (n_behaviours, n_runs, n_samples, num_steps)  float64, NaN where absent
    flops  : (n_behaviours, n_runs, num_steps)             float64

    n_samples is fixed at 50 (the hard cap in run.json); steps where fewer samples
    were drawn have NaN in the trailing slots.  Steps skipped by sampling_schedule
    are fully NaN across all sample slots.

    Parameters
    ----------
    num_steps : int or None
        Number of steps to allocate. When None (default), inferred from the run
        data itself (see _infer_num_steps) rather than from a config value, so a
        resumed/continued run isn't silently truncated.
    num_behaviours : int or None
        Expected numbered subfolders per run.  If a folder has more (e.g. from a
        parallel-submission timestamp collision), its subfolders are automatically
        split into groups by attack seed.  When None, uses the CSV column value.
    filter_seed : int or None
        If given, only keep the run(s) whose attack seed equals this value (n_runs
        shrinks accordingly). Prints a warning if that's not exactly one run.

    A seed that's missing some behaviours (e.g. a failed job) is not dropped: its
    missing (behaviour, run) slots are simply left as NaN in scores and 0 in flops,
    matching how a single unreadable run.json is already handled below. A warning
    listing the affected idx and how many seeds cover them is printed by
    _build_virtual_runs when this happens.
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_beh = (
        num_behaviours
        if num_behaviours is not None
        else int(rows[0].get("num_numbered_folders", 1))
    )
    virtual_runs, run_seeds = _build_virtual_runs(rows, n_beh, base_path)

    if filter_seed is not None:
        keep = [i for i, seed in enumerate(run_seeds) if seed == filter_seed]
        if len(keep) != 1:
            print(
                f"Warning: filter_seed={filter_seed} matched {len(keep)} run(s), expected exactly 1."
            )
        virtual_runs = [virtual_runs[i] for i in keep]
        run_seeds = [run_seeds[i] for i in keep]

    if num_steps is None:
        num_steps = _infer_num_steps(virtual_runs, base_path)
        print(f"Inferred num_steps={num_steps} from run data (last recorded step + 1)")

    n_runs = len(virtual_runs)
    scores = np.full((n_beh, n_runs, num_samples, num_steps), np.nan)
    flops = np.zeros((n_beh, n_runs, num_steps))

    print(f"Loading {n_runs} run(s) × {n_beh} behaviour(s) from {csv_path}")
    for r_idx, run_entries in enumerate(tqdm(virtual_runs, desc="Loading runs")):
        for b_idx, entry in enumerate(run_entries):
            if entry is None:
                continue
            full_path, subfolder_idx = entry
            path = os.path.join(base_path, full_path, str(subfolder_idx), "run.json")
            step_scores, step_flops, _ = _load_run_with_flops(path, num_steps, num_samples)
            if step_scores is not None:
                scores[b_idx, r_idx] = (
                    step_scores.T
                )  # (num_steps, MAX_SAMPLES_PER_STEP).T → (MAX_SAMPLES_PER_STEP, num_steps)
                flops[b_idx, r_idx] = step_flops

                if b_idx == 90 and r_idx == 0:
                    print(
                        f"Scores for behaviour {b_idx}, run {r_idx}: {scores[b_idx, r_idx][:10]}, {scores[b_idx, r_idx].shape}"
                    )

    print(f"Loaded: scores={scores.shape}, flops={flops.shape}")
    return scores, flops


def csv_stem(csv_path: str) -> str:
    return os.path.splitext(os.path.basename(os.path.normpath(csv_path)))[0]


def _scores_fingerprint(
    csv_path: str,
    base_path: str,
    filter_seed: int | None,
    num_samples: int,
    num_steps: int | None,
    num_behaviours: int | None,
) -> dict:
    st = os.stat(csv_path)
    return {
        "csv_path": os.path.abspath(csv_path),
        "csv_mtime": st.st_mtime,
        "csv_size": st.st_size,
        "base_path": base_path,
        "filter_seed": filter_seed,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "num_behaviours": num_behaviours,
    }


def save_scores(
    csv_path: str,
    scores_dir: str = SCORES_DIR,
    base_path: str = OUTPUTS_BASE_PATH,
    num_samples: int = 32,
    num_steps: int | None = None,
    num_behaviours: int | None = None,
    filter_seed: int | None = None,
    force: bool = False,
) -> str:
    """
    Load `csv_path` via `load_experiment` and cache the resulting
    `all_scores`/`all_flops` arrays under `<scores_dir>/<csv_stem>/` as
    `all_scores.npy` / `all_flops.npy`, alongside a `meta.json` fingerprint.

    A cache hit (matching `meta.json`) is reused as-is unless `force=True`.
    Returns the cache directory path.
    """
    stem = csv_stem(csv_path)
    cache_dir = os.path.join(scores_dir, stem)
    meta_path = os.path.join(cache_dir, "meta.json")
    fingerprint = _scores_fingerprint(
        csv_path, base_path, filter_seed, num_samples, num_steps, num_behaviours
    )

    if not force and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if all(meta.get(k) == v for k, v in fingerprint.items()):
            print(f"Cache hit: {cache_dir}/ is already up to date; nothing to do.")
            return cache_dir
        print(
            f"Cache at {cache_dir}/ is stale (source CSV or load parameters changed); recomputing."
        )

    all_scores, all_flops = load_experiment(
        csv_path,
        num_samples=num_samples,
        num_steps=num_steps,
        num_behaviours=num_behaviours,
        base_path=base_path,
        filter_seed=filter_seed,
    )

    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "all_scores.npy"), all_scores)
    np.save(os.path.join(cache_dir, "all_flops.npy"), all_flops)

    meta = {
        **fingerprint,
        "all_scores_shape": list(all_scores.shape),
        "all_flops_shape": list(all_flops.shape),
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved scores to {cache_dir}/")
    return cache_dir


def _flatten_scores(scores):
    """Convert load_experiment's (n_behaviours, n_runs, n_samples, num_steps) array
    into the legacy per-behaviour list format: n_behaviours arrays of shape
    (num_steps, n_runs * n_samples). No-op if already given in the legacy format.
    """
    if isinstance(scores, np.ndarray) and scores.ndim == 4:
        n_beh, n_runs, n_samples, num_steps = scores.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return [scores[b].reshape(n_runs * n_samples, num_steps).T for b in range(n_beh)]
    return scores


def _flatten_flops(flops):
    """Convert load_experiment's (n_behaviours, n_runs, num_steps) array into the
    legacy per-behaviour list format: n_behaviours arrays of shape (num_steps,),
    averaged over runs. No-op if already given in the legacy format.
    """
    if isinstance(flops, np.ndarray) and flops.ndim == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return [np.nanmean(flops[b], axis=0) for b in range(flops.shape[0])]
    return flops


def load_scores_merged_runs(
    csv_path,
    num_steps=None,
    num_samples=32,
    num_behaviours=None,
    base_path=OUTPUTS_BASE_PATH,
    filter_seed=None,
):
    """Backward-compatible wrapper around load_experiment.

    Returns (all_scores, all_flops) in the legacy flat format:
      all_scores : list of (num_steps, n_runs * n_samples) arrays, one per behaviour
      all_flops  : list of (num_steps,) arrays (mean over runs), one per behaviour
    """
    scores, flops = load_experiment(
        csv_path,
        num_samples=num_samples,
        num_steps=num_steps,
        num_behaviours=num_behaviours,
        base_path=base_path,
        filter_seed=filter_seed,
    )
    return _flatten_scores(scores), _flatten_flops(flops)


def _ffill_rows(y):
    """Forward-fill NaNs along axis=1 (time), independently per row of y (n_lines, n_steps).

    A row's leading NaNs (before its own first valid value) are left as NaN --
    there is no earlier value to carry forward there.
    """
    y = np.asarray(y, dtype=float)
    valid_idx = np.where(np.isnan(y), -1, np.arange(y.shape[1]))
    last_valid_idx = np.maximum.accumulate(valid_idx, axis=1)
    rows = np.arange(y.shape[0])[:, None]
    return np.where(last_valid_idx == -1, np.nan, y[rows, np.clip(last_valid_idx, 0, None)])


def _plot_overlaid_lines(
    ax,
    y,
    colors,
    alpha,
    mean_color,
    category_legend=None,
    show_legend=True,
    legend_fontsize=None,
    legend_loc="best",
):
    """Plot each row of y (n_lines, n_steps) as a faint line, plus a bold nanmean-of-rows line.

    category_legend, if given, is a list of (label, color) pairs added to the
    legend as proxy handles -- used when `colors` assigns colors by category
    membership rather than one color per individual line, so the category
    names still show up in the legend instead of just the mean line. Underscores in every
    legend label (mean_label included) are shown as spaces.

    show_legend : draw the legend on this axes at all -- False for panels that share a legend
    drawn elsewhere (e.g. a multi-panel comparison where only the leftmost panel needs one).
    legend_fontsize : font size for the legend specifically; None keeps whatever the current
    rcParams say (useful when the rest of the axes' text has been scaled up via `rc_context`
    but the legend should stay at the unscaled size).
    """
    has_data = ~np.all(np.isnan(y), axis=0)
    x = np.where(has_data)[0]

    for i in range(y.shape[0]):
        color = colors[i] if isinstance(colors, list) else colors
        ax.plot(x, y[i, has_data], color=color, alpha=alpha, linewidth=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(y[:, has_data], axis=0)
    ax.plot(x, mean, color=mean_color, linewidth=2.0)

    if category_legend:
        for label, color in category_legend:
            ax.plot([], [], color=color, linewidth=2.0, label=label.replace("_", " "))

    ax.set_xlabel("Step")
    if show_legend:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    ax.grid(True, alpha=0.3)


def _category_colors(n_lines, categories):
    """Map each of n_lines row indices [0, n_lines) to a color by category membership.

    categories : dict[str, list[int]] mapping a category name to the row
    indices belonging to it. An index absent from every category gets a
    fixed neutral gray rather than being silently miscolored. Returns
    (colors, legend): colors is a list of n_lines colors (one per line);
    legend is a list of (label, color) pairs -- one per category present,
    plus "uncategorized" if any index wasn't covered.
    """
    names = list(categories)
    cmap = plt.cm.tab10 if len(names) <= 10 else plt.cm.tab20
    cat_colors = {name: cmap(i / max(len(names) - 1, 1)) for i, name in enumerate(names)}

    idx_to_category = {}
    for name, indices in categories.items():
        for idx in indices:
            idx_to_category[idx] = name

    uncategorized_color = "#b0b0b0"
    colors = [cat_colors.get(idx_to_category.get(i), uncategorized_color) for i in range(n_lines)]

    legend = [(name, cat_colors[name]) for name in names]
    if any(i not in idx_to_category for i in range(n_lines)):
        legend.append(("uncategorized", uncategorized_color))
    return colors, legend


def _viridis_category_colors(n_lines, categories):
    """`_category_colors`, but samples the shared `viridis` colormap instead of tab10/tab20 --
    for functions that have opted into the house palette established by
    `plot_asr_decomposition`/`plot_asr_compute_heatmap` in plot_forecasts.py (viridis for
    per-item color, black for an aggregate/mean line). Kept as a separate function rather than
    changing `_category_colors` itself, since that one is still used by
    `plot_success_prob_forecast_with_residuals`, which hasn't opted into this palette -- forking
    here means that plot's colors don't change as a side effect of styling the others.
    """
    names = list(categories)
    cat_colors = {
        name: plt.cm.viridis(i / max(len(names) - 1, 1) * _VIRIDIS_LINE_MAX)
        for i, name in enumerate(names)
    }

    idx_to_category = {}
    for name, indices in categories.items():
        for idx in indices:
            idx_to_category[idx] = name

    uncategorized_color = "#b0b0b0"
    colors = [cat_colors.get(idx_to_category.get(i), uncategorized_color) for i in range(n_lines)]

    legend = [(name, cat_colors[name]) for name in names]
    if any(i not in idx_to_category for i in range(n_lines)):
        legend.append(("uncategorized", uncategorized_color))
    return colors, legend


def plot_overlaid_runs(
    scores,
    title=None,
    y_label="Mean Strong Reject Score (p_harmful)",
    save=True,
    categories=None,
    xscale="linear",
    yscale="linear",
):
    """One panel per array in `scores`, side by side: each panel plots one faint line per row
    of that array plus a bold mean-of-rows curve, optionally colored/labeled by category.

    scores : list of plain (n_lines, n_steps) arrays, e.g. `[scores_llama, scores_gemma]` for
    a side-by-side model comparison -- one panel per array, in order. Wrap a single array as
    `[scores]` for one panel.

    title : a single string (used as the figure's overall suptitle), a list of per-panel
    titles (same length as `scores`), or None for no title(s).

    y_label : shown only on the leftmost panel, since it applies to every panel alike.

    categories : optional dict[str, list[int]] mapping a category name to row indices, applied
    identically to every panel (every array in `scores` must therefore share the same row
    indexing). When given, lines are colored by category membership instead of the default
    per-line rainbow. Only the leftmost panel gets a legend, to avoid repeating it in every
    panel; underscores in category names are shown as spaces there.

    xscale, yscale : passed straight to Axes.set_xscale/set_yscale (e.g. "log"), applied to
    every panel. Non-positive values (x=0, or a score of 0.0) can't be shown on a log axis --
    matplotlib silently drops those points rather than raising, so a log-scaled line may start
    later or have gaps compared to the linear version.

    Title/axis-label/tick text is drawn at 1.5x the current default font size; the legend is
    kept at the unscaled size so it doesn't dominate the panel.
    """
    if not isinstance(scores, (list, tuple)):
        raise TypeError(
            "`scores` must be a list of (n_lines, n_steps) arrays -- wrap a single array as [scores]."
        )
    scores = [np.asarray(s) for s in scores]
    n_panels = len(scores)

    if isinstance(title, (list, tuple)):
        panel_titles, suptitle = list(title), None
    else:
        panel_titles, suptitle = [None] * n_panels, title

    with plt.rc_context({**_LATEX_FONT_RC, **_SLEEK_EXTRA_RC, "font.size": _BIG_FONT}):
        fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 7), sharey=True)
        axes = np.atleast_1d(axes)

        for i, (ax, s) in enumerate(zip(axes, scores, strict=True)):
            n_lines = s.shape[0]
            if categories is not None:
                colors, legend = _viridis_category_colors(n_lines, categories)
            else:
                colors = [
                    plt.cm.viridis(j / max(n_lines - 1, 1) * _VIRIDIS_LINE_MAX)
                    for j in range(n_lines)
                ]
                legend = None

            _plot_overlaid_lines(
                ax,
                s,
                colors,
                0.4,
                "black",
                category_legend=legend,
                show_legend=(i == 0),
                legend_fontsize=_BIG_LEGEND_FONT,
                legend_loc="upper left",
            )
            if panel_titles[i]:
                ax.set_title(panel_titles[i])
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            # Override `_plot_overlaid_lines`'s plain default grid with the same refined,
            # open-spined look used across the other thesis figures (plot_asr_at_k_fit,
            # plot_asr_decomposition, etc.), so this figure reads as part of the same family
            # rather than plain default matplotlib styling.
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
            ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
            ax.tick_params(direction="out", length=5)

        axes[0].set_ylabel(y_label)

        fig.tight_layout(pad=1.4, w_pad=0.6)

        if suptitle:
            fig.tight_layout(rect=[0, 0, 1, 0.955], pad=1.4, w_pad=0.6)
            fig.suptitle(suptitle, y=0.98)

        plt.show()

    _save_fig(fig, suptitle, "overlaid_runs", save)


def plot_asr_at_threshold(
    all_scores, threshold, sample_count=32, aggregate_only=False, title=None, save=False
):
    """Plot the change in ASR@threshold (relative to each run's first step
    with data) per step, per run, with a bold mean-of-runs curve.

    all_scores : (n_behaviours, n_runs, n_samples, num_steps) array.
    At each step, a behaviour counts as broken for a run if at least one of
    its samples (NaN samples ignored) has a score >= `threshold`. A run's
    ASR at that step is the fraction of behaviours broken, with behaviours
    that have no valid (non-NaN) sample at that step/run excluded from both
    numerator and denominator. Each run's ASR curve is then re-based to its
    own first non-NaN value (judge scores are often only logged starting a
    few steps in, so literal step 0 may have no data at all), so every line
    starts at 0.
    One faint line per run, plus a bold mean-of-runs line.

    aggregate_only=True draws only the mean-of-runs line (no per-run lines)
    and auto-scales the y-axis to that line's own range, since the mean
    usually occupies only a narrow band of the full [-1, 1] range used when
    every run is shown alongside it.
    """
    valid = ~np.isnan(all_scores)  # (n_beh, n_runs, n_samples, n_steps)
    crosses = (all_scores > threshold) & valid
    broken = crosses.any(axis=2)  # (n_beh, n_runs, n_steps)
    beh_valid = valid.any(axis=2)  # (n_beh, n_runs, n_steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        asr = broken.sum(axis=0) / beh_valid.sum(axis=0)  # (n_runs, n_steps)

        # Re-base each run to its own first non-NaN value, since literal
        # step 0 is often entirely unlogged.
        has_valid = ~np.all(np.isnan(asr), axis=1)
        first_idx = np.isnan(asr).argmin(axis=1)  # first False (=non-NaN) index per row
        baseline = np.where(has_valid, asr[np.arange(asr.shape[0]), first_idx], np.nan)
        asr = asr - baseline[:, None]

    n_runs = asr.shape[0]
    fig, ax = plt.subplots(figsize=(9, 5))

    if aggregate_only:
        has_data = ~np.all(np.isnan(asr), axis=0)
        x = np.where(has_data)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(asr[:, has_data], axis=0)
        ax.plot(x, mean, color="black", linewidth=2.0, label=f"mean ({n_runs} runs)")
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

        finite_mean = mean[np.isfinite(mean)]
        if len(finite_mean):
            lo, hi = float(finite_mean.min()), float(finite_mean.max())
            pad = max((hi - lo) * 0.1, 0.01)
            ax.set_ylim(lo - pad, hi + pad)
    else:
        cmap = plt.cm.tab20 if n_runs <= 20 else plt.cm.viridis
        colors = [cmap(i / max(n_runs - 1, 1)) for i in range(n_runs)]
        _plot_overlaid_lines(ax, asr, colors, 0.4, "black", f"mean ({n_runs} runs)")

        finite_asr = asr[np.isfinite(asr)]
        if len(finite_asr):
            lo, hi = float(finite_asr.min()), float(finite_asr.max())
            pad = max((hi - lo) * 0.1, 0.01)
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylabel(f"ΔASR@{threshold:g} (rel. to step 0)")
    ax.set_title((title or "") + f" – ASR@{sample_count:g}, {n_runs} runs")
    fig.tight_layout()

    if save and title:
        fig.savefig(f"{title.replace(' ', '_')}_asr_at_{threshold:g}.png", dpi=150)

    plt.show()


def plot_variance_behaviours(scores, title=None, save=False):
    """Plot sample variance per step for each behaviour overlaid, with a bold grand variance.

    scores : (n_behaviours, n_runs, n_samples, num_steps) from load_experiment.
    Each faint line is the variance of all runs×samples for one behaviour at each step.
    The bold black line is the variance pooled over all behaviours, runs, and samples.
    """
    n_beh, n_runs, _, n_steps = scores.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beh_vars = np.nanvar(scores, axis=(1, 2))  # (n_beh, n_steps)

    has_data = ~np.all(np.isnan(beh_vars), axis=0)
    x = np.where(has_data)[0]

    cmap = plt.cm.tab20 if n_beh <= 20 else plt.cm.viridis
    colors = [cmap(i / max(n_beh - 1, 1)) for i in range(n_beh)]

    fig, ax = plt.subplots(figsize=(11, 5))
    for b_idx, color in enumerate(colors):
        ax.plot(x, beh_vars[b_idx, has_data], color=color, alpha=0.4, linewidth=0.8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grand_var = np.nanvar(scores[:, :, :, has_data], axis=(0, 1, 2))
    ax.plot(
        x, grand_var, color="black", linewidth=2.0, label=f"variance pooled ({n_beh} behaviours)"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Variance of Strong Reject Score (p_harmful)")
    ax.set_title((title or "") + f" – sample variance, {n_beh} behaviours overlaid")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save and title:
        fig.savefig(f"{title.replace(' ', '_')}_variance_behaviours.png", dpi=150)

    plt.show()


def plot_experiment(
    csv_path=None,
    num_steps=None,
    title=None,
    save=False,
    filters=None,
    all_scores=None,
    all_flops=None,
    behaviour_ids=None,
):
    """Load data, compute metrics, and plot for one experiment.

    Pass all_scores (and optionally all_flops) directly to skip CSV loading.
    filters: optional dict of column->value pairs for parquet files.
    behaviour_ids: optional list of dataset idx values to restrict which behaviours are loaded.
    """
    if all_scores is None:
        if _is_parquet(csv_path):
            all_scores = load_scores(csv_path, num_steps, filters=filters)
            all_flops = None
        else:
            all_scores, all_flops, _ = _load_scores_csv_with_flops(
                csv_path, num_steps, behaviour_ids=behaviour_ids
            )

    if not all_scores:
        print("No valid data found, skipping plot.")
        return

    if all_flops is not None:
        stacked_flops = np.stack(all_flops, axis=0)
        has_flops = np.any(stacked_flops > 0, axis=1)
        if has_flops.any():
            cum_flops = np.cumsum(stacked_flops[has_flops], axis=1)
            x_values = np.mean(cum_flops, axis=0) / 10000000
            xlabel = "Cumulative FLOPs (TFLOPs)"
        else:
            x_values = np.arange(num_steps or len(all_scores[0]))
            xlabel = "Step"
    else:
        x_values = np.arange(num_steps or len(all_scores[0]))
        xlabel = "Step"

    mean_avg, mean_max, mean_1st, mean_cummax_1st, mean_var = compute_metrics(all_scores)

    # With a sampling schedule most steps have no data; drop them so lines are continuous
    sampled = ~np.isnan(mean_avg)
    if not sampled.all():
        x_values = x_values[sampled]
        mean_avg = mean_avg[sampled]
        mean_max = mean_max[sampled]
        # mean_1st        = mean_1st[sampled]
        mean_cummax_1st = mean_cummax_1st[sampled]
        mean_var = mean_var[sampled]

    print(f"mean_avg        : min={mean_avg.min():.4f}, max={mean_avg.max():.4f}")
    # print(f"mean_max        : min={mean_max.min():.4f}, max={mean_max.max():.4f}")
    print(f"mean_1st        : min={mean_1st.min():.4f}, max={mean_1st.max():.4f}")
    print(f"mean_cummax_1st : min={mean_cummax_1st.min():.4f}, max={mean_cummax_1st.max():.4f}")
    print(f"mean_var        : min={mean_var.min():.4f}, max={mean_var.max():.4f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_values, mean_avg, label="Mean of samples (avg over behaviours)", linewidth=1.5)
    # ax.plot(x_values, mean_max,        label="Max of samples (avg over behaviours)",                     linewidth=1.5)
    # ax.plot(x_values, mean_1st,        label="1st sample, greedy (avg over behaviours)",                 linewidth=1.5)
    # ax.plot(x_values, mean_cummax_1st, label="Cumulative max of 1st sample, greedy (avg over behav)",    linewidth=1.5, linestyle="--")
    # ax.plot(x_values, mean_var,        label="Variance of samples (avg over behaviours)",                linewidth=1.5, linestyle=":")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Strong Reject Score")
    ax.set_title(title or (os.path.basename(csv_path) if csv_path else ""))
    ax.legend()
    ax.set_xlim(x_values[0], x_values[-1])
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(0, 0.2)
    fig.tight_layout()

    if save and csv_path:
        out = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "harmfulness_plot.png")
        fig.savefig(out, dpi=150)
        print(f"Plot saved to {out}")

    plt.show()


def _load_run_with_flops(run_json_path, num_steps, num_samples, behaviour_ids=None):
    """Read one run.json; return ((num_steps, num_samples) scores, (num_steps,) flops, prompt str) or (None, None, None)."""
    try:
        with open(run_json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: {run_json_path}: {e}")
        return None, None, None

    if behaviour_ids is not None:
        try:
            raw_idx = data["config"]["dataset_params"]["idx"]
            idx = raw_idx[0] if isinstance(raw_idx, list) else raw_idx
            if int(idx) not in behaviour_ids:
                return None, None, None
        except (KeyError, TypeError, ValueError, IndexError):
            pass  # can't determine idx; include the run

    runs = data.get("runs", [])
    if not runs:
        print(f"  Warning: no 'runs' key in {run_json_path}")
        return None, None, None

    try:
        prompt = runs[0]["original_prompt"][0]["content"]
    except (KeyError, IndexError, TypeError):
        prompt = ""

    step_list = runs[0].get("steps", [])
    scores = np.full((num_steps, num_samples), np.nan)
    flops = np.zeros(num_steps)
    for step_data in step_list:
        idx = step_data.get("step")
        if idx is None or idx >= num_steps:
            continue
        flops[idx] = step_data.get("flops", 0)
        try:
            p = step_data["scores"][SCORE_KEY][METRIC_KEY]
            n = min(len(p), num_samples)
            scores[idx, :n] = p[:n]
        except (KeyError, TypeError):
            pass
    return scores, flops, prompt


def _load_scores_csv_with_flops(csv_path, num_steps, behaviour_ids=None):
    """CSV-based loader that also returns per-step FLOPs and SemanticCategory per behaviour.

    Returns (all_scores, all_flops, all_categories).
    Categories are looked up by matching the run's prompt against the harmbench Behavior
    column using endswith (prompt from run.json ends with the Behavior string).
    behaviour_ids: optional list of dataset idx values to restrict which behaviours are loaded.
    """
    harmbench = []
    with open(HARMBENCH_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            harmbench.append((row["Behavior"], row["SemanticCategory"]))

    def _find_category(prompt):
        for behavior, cat in harmbench:
            if prompt.endswith(behavior):
                return cat
        return "unknown"

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} row(s) from {csv_path}")

    all_scores = []
    all_flops = []
    all_categories = []
    for row in tqdm(rows, desc="Loading run.json files"):
        full_path = row["full_path"].strip()
        if full_path.endswith(".json"):
            paths = [os.path.join(BASE_PATH, full_path)]
        else:
            num_folders = int(row.get("num_numbered_folders", 1) or 1)
            paths = [
                os.path.join(BASE_PATH, full_path, str(i), "run.json") for i in range(num_folders)
            ]
        for path in paths:
            scores, flops, prompt = _load_run_with_flops(
                path, num_steps, behaviour_ids=behaviour_ids
            )
            if scores is not None:
                all_scores.append(scores)
                all_flops.append(flops)
                all_categories.append(_find_category(prompt))

    print(f"Successfully loaded {len(all_scores)} / {len(rows)} file(s).")
    return all_scores, all_flops, all_categories


def plot_score_vs_flops(
    all_scores,
    all_flops,
    csv_path,
    num_steps,
    steps_to_plot,
    flops_per_sample_T=1200000000000,
    title=None,
    save=False,
    filters=None,
):
    """Plot avg strongreject score vs cumulative TFLOPs for selected evaluation steps.

    steps_to_plot: e.g. [1, 5, 10, 20, 40]. Each step becomes one line.
    The x-axis anchor for step k is the cumulative FLOPs (summed from step 0 to k).
    Within the step, sample i sits at anchor + i * flops_per_sample_T TFLOPs.
    The y-value is the avg strongreject score at that sample position across all behaviours.

    Only supported for CSV / run.json data (parquet files don't carry per-step FLOPs).
    """
    # if _is_parquet(csv_path):
    #     print("Warning: per-step FLOPs not available in parquet files; skipping this plot.")
    #     return
    # if filters:
    #     print("Warning: filters are only supported for parquet files and will be ignored.")

    # all_scores, all_flops = _load_scores_csv_with_flops(csv_path, num_steps)
    all_scores = _flatten_scores(all_scores)
    all_flops = _flatten_flops(all_flops)

    if not all_scores:
        print("No valid data found, skipping plot.")
        return

    stacked_scores = np.stack(all_scores, axis=0)  # (n_files, num_steps, 50)
    stacked_flops = np.stack(all_flops, axis=0)  # (n_files, num_steps), raw FLOPs

    if np.all(stacked_flops == 0):
        print("Warning: all flops values are 0 – FLOPs data not recorded in these runs.")
        return

    # Use only files that have at least some non-zero FLOPs recorded
    has_flops = np.any(stacked_flops > 0, axis=1)
    cum_T = np.cumsum(stacked_flops[has_flops], axis=1)  # FLOPs, (n_valid, num_steps)
    mean_cum_T = np.mean(cum_T, axis=0) / 10  # (num_steps,)

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.viridis
    n_lines = len(steps_to_plot)
    colors = [cmap(i / max(n_lines - 1, 1)) for i in range(n_lines)]

    n_samples = stacked_scores.shape[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for color, step_idx in zip(colors, steps_to_plot):
            if step_idx >= num_steps:
                print(f"  Warning: step {step_idx} >= num_steps {num_steps}, skipping.")
                continue
            # flops_per_sample_T *= 100   # So that curves have some shape - sampling FLOPs are almost negligible compared to opt FLOPs
            x = mean_cum_T[step_idx] + np.arange(n_samples) * flops_per_sample_T
            step_scores = stacked_scores[:, step_idx, :]  # (n_files, n_samples)
            # Running best across sample positions; NaN → -inf for accumulate, then restore
            finite = np.where(np.isnan(step_scores), -np.inf, step_scores)
            cummax = np.maximum.accumulate(finite, axis=1)
            cummax = np.where(np.isneginf(cummax), np.nan, cummax)
            y = np.nanmean(cummax, axis=0)  # (n_samples,) avg best-so-far over behaviours
            n = step_idx + 1
            if 11 <= (n % 100) <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            ax.plot(x, y, label=f"{n}{suffix} step", color=color, linewidth=1.5)

    ax.set_xlabel("Optimisation FLOPs (TFLOPs)")
    ax.set_ylabel("Avg Best-So-Far Strong Reject Score (p_harmful)")
    ax.set_title((title or os.path.basename(csv_path)) + " – score vs FLOPs by step")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # if save:
    #     out = os.path.join(
    #         os.path.dirname(os.path.abspath(csv_path)),
    #         "score_vs_flops.png",
    #     )
    #     fig.savefig(out, dpi=150)
    #     print(f"Plot saved to {out}")

    plt.show()


def plot_score_vs_flops_by_category(
    all_scores,
    all_flops,
    categories,
    num_steps,
    steps_to_plot,
    flops_per_sample_T=1200000000000,
    title=None,
    save=False,
):
    """Like plot_score_vs_flops but creates one subplot per SemanticCategory, stacked vertically.

    categories: list of SemanticCategory strings, parallel to all_scores / all_flops.
    All other parameters are forwarded to the per-category plot identically.
    """
    all_scores = _flatten_scores(all_scores)
    all_flops = _flatten_flops(all_flops)

    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)
    if n_cats == 0:
        print("No categories found, skipping plot.")
        return

    fig, axes = plt.subplots(n_cats, 1, figsize=(11, 5 * n_cats), squeeze=False)

    cmap = plt.cm.viridis
    n_lines = len(steps_to_plot)
    line_colors = [cmap(i / max(n_lines - 1, 1)) for i in range(n_lines)]

    for ax_idx, cat in enumerate(unique_cats):
        ax = axes[ax_idx, 0]
        indices = [i for i, c in enumerate(categories) if c == cat]
        cat_scores = [all_scores[i] for i in indices]
        cat_flops = [all_flops[i] for i in indices]

        if not cat_scores:
            ax.set_title(f"{cat} (no data)")
            continue

        stacked_scores = np.stack(cat_scores, axis=0)
        stacked_flops = np.stack(cat_flops, axis=0)

        if np.all(stacked_flops == 0):
            ax.set_title(f"{cat} (no FLOPs data, n={len(cat_scores)})")
            continue

        has_flops = np.any(stacked_flops > 0, axis=1)
        cum_T = np.cumsum(stacked_flops[has_flops], axis=1)
        mean_cum_T = np.mean(cum_T, axis=0) / 10

        n_samples = stacked_scores.shape[2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for color, step_idx in zip(line_colors, steps_to_plot):
                if step_idx >= num_steps:
                    continue
                x = mean_cum_T[step_idx] + np.arange(n_samples) * flops_per_sample_T
                step_scores = stacked_scores[:, step_idx, :]
                finite = np.where(np.isnan(step_scores), -np.inf, step_scores)
                cummax = np.maximum.accumulate(finite, axis=1)
                cummax = np.where(np.isneginf(cummax), np.nan, cummax)
                y = np.nanmean(cummax, axis=0)
                n = step_idx + 1
                suffix = (
                    "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                )
                ax.plot(x, y, label=f"{n}{suffix} step", color=color, linewidth=1.5)

        ax.set_title(f"{cat}  (n={len(cat_scores)})")
        ax.set_xlabel("Optimisation FLOPs (TFLOPs)")
        ax.set_ylabel("Avg Best-So-Far Score")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout()

    if save and title:
        out = os.path.join(
            os.path.dirname(CATEGORY_LOOKUP_PATH), f"{title}_score_vs_flops_by_category.png"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out}")

    plt.show()


def plot_score_distributions(
    title,
    csv_path=None,
    num_steps=None,
    n_distributions=10,
    steps=None,
    save=False,
    filters=None,
    semantic_category="all",
    categories=None,
    all_scores=None,
):
    """Plot the distribution of harmfulness scores across samples for n_distributions
    evenly-spaced steps, shown as three side-by-side columns:
      - "average": nanmean over behaviours → (50,) per step
      - "pooled":  all behaviours flattened → (n_files * 50,) per step
      - "single":  first behaviour only → (50,) per step

    Pass all_scores (and optionally categories) directly to skip CSV loading.
    """
    from scipy.stats import gaussian_kde

    if all_scores is None:
        all_scores = load_scores(csv_path, num_steps, filters=filters)
    all_scores = _flatten_scores(all_scores)

    if not all_scores:
        print("No valid data found, skipping plot.")
        return

    if semantic_category != "all" and categories is not None:
        all_scores = [s for s, c in zip(all_scores, categories) if c == semantic_category]
        if not all_scores:
            print(f"No data for semantic_category={semantic_category!r}, skipping plot.")
            return

    stacked = np.stack(all_scores, axis=0)  # (n_files, num_steps, 50)

    if steps is not None:
        step_indices = list(steps)
    elif n_distributions == 1:
        step_indices = [0]
    else:
        step_indices = [
            round(i * (num_steps - 1) / (n_distributions - 1)) for i in range(n_distributions)
        ]

    aggregations = [
        ("average", lambda s: np.nanmean(stacked[:, s, :], axis=0)),
        ("pooled", lambda s: stacked[:, s, :].ravel()),
        ("single", lambda s: stacked[0, s, :]),
    ]

    # ── Layout ───────────────────────────────────────────────────────────────
    row_h = 1.1
    fig_h = row_h * n_distributions + 1.0
    fig_w = 7.0
    n_cols = len(aggregations)

    fig, axes = plt.subplots(
        n_distributions,
        n_cols,
        figsize=(fig_w * n_cols, fig_h),
        sharex=True,
        sharey=False,
    )
    if n_distributions == 1:
        axes = axes[np.newaxis, :]

    x_grid = np.linspace(0, 1, 400)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_distributions - 1, 1)) for i in range(n_distributions)]

    for col_idx, (agg_name, agg_fn) in enumerate(aggregations):
        axes[0, col_idx].set_title(agg_name, fontsize=10, fontweight="bold", pad=6)

        for row_idx, (s, color) in enumerate(zip(step_indices, colors)):
            ax = axes[row_idx, col_idx]

            vals = agg_fn(s)
            vals = vals[~np.isnan(vals)]
            mu = float(np.mean(vals)) if len(vals) else float("nan")
            var = float(np.var(vals)) if len(vals) else float("nan")

            if len(vals) >= 2:
                try:
                    kde = gaussian_kde(vals, bw_method="scott")
                    density = kde(x_grid)
                    ax.fill_between(x_grid, density, alpha=0.35, color=color)
                    ax.plot(x_grid, density, color=color, linewidth=1.2)
                except np.linalg.LinAlgError:
                    pass

            jitter = np.random.default_rng(s).uniform(-0.015, 0.015, size=len(vals))
            ax.plot(vals, jitter, "|", color=color, alpha=0.25, markersize=4)

            if col_idx == 0:
                ax.set_ylabel(f"step {s}", fontsize=7, rotation=0, labelpad=40, va="center")
            ax.yaxis.set_label_position("left")
            ax.set_yticks([])

            ax.text(
                0.97,
                0.95,
                f"μ={mu:.3f}\nσ²={var:.4f}",
                transform=ax.transAxes,
                fontsize=6.5,
                va="top",
                ha="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, lw=0),
            )

            ax.set_xlim(0, 1)
            ax.spines[["top", "right", "left"]].set_visible(False)

        axes[-1, col_idx].set_xlabel("Strong Reject Score (p_harmful)", fontsize=8)

    fig.suptitle(
        title,
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    if save:
        out = os.path.join(
            os.path.dirname(os.path.abspath(csv_path)),
            "harmfulness_distributions.png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out}")

    plt.show()


def plot_behaviour_score_distributions(
    title,
    all_scores,
    success_prob=False,
    n_panels=4,
    steps=None,
    save=False,
):
    """Plot, side by side, the distribution across behaviours of each behaviour's
    mean judge score (nanmean over its runs and samples) at `n_panels` evenly-spaced
    training steps.

    Each panel is one step: a gaussian KDE + filled density over the n_behaviours
    per-behaviour means, with a rug of the individual behaviour means underneath
    and a small n / mu / sigma annotation -- the same visual language as
    `plot_score_distributions`, but one point per behaviour instead of per sample.

    all_scores : either the raw (n_behaviours, n_runs, n_samples, num_steps) array
    from load_experiment, or the legacy per-behaviour list format (n_behaviours
    arrays of shape (num_steps, n_runs * n_samples)) -- both accepted via
    _flatten_scores.

    steps : explicit list of step (column) indices to show; default is n_panels
    evenly-spaced steps chosen from the steps that actually have data (scores
    are typically logged only every ~10 steps, so evenly spacing over the raw
    [0, num_steps) range would mostly land on empty columns).
    """
    from scipy.stats import gaussian_kde

    all_scores = _flatten_scores(all_scores)
    if not all_scores:
        print("No valid data found, skipping plot.")
        return

    stacked = np.stack(all_scores, axis=0)  # (n_behaviours, num_steps, n_runs * n_samples)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beh_means = np.nanmean(stacked, axis=2)  # (n_behaviours, num_steps)

        if success_prob:
            success = np.where(np.isnan(stacked), np.nan, (stacked > 0.8).astype(float))
            # Per behaviour/step fraction of samples with score > 0.5
            plot_vals = np.nanmean(success, axis=2)  # (n_behaviours, num_steps)
        else:
            plot_vals = beh_means

    if steps is not None:
        step_indices = list(steps)
    else:
        valid_steps = np.flatnonzero(np.any(~np.isnan(beh_means), axis=0))
        if len(valid_steps) == 0:
            print("plot_behaviour_score_distributions: no non-NaN data found, skipping plot.")
            return
        if n_panels == 1 or len(valid_steps) == 1:
            step_indices = [int(valid_steps[0])]
        else:
            positions = [
                round(i * (len(valid_steps) - 1) / (n_panels - 1)) for i in range(n_panels)
            ]
            step_indices = [int(valid_steps[p]) for p in positions]

    n_panels = len(step_indices)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 3.6), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    x_grid = np.linspace(0, 1, 400)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_panels - 1, 1)) for i in range(n_panels)]

    for ax, color, s in zip(axes, colors, step_indices):
        vals = plot_vals[:, s]
        vals = vals[~np.isnan(vals)]
        mu = float(np.mean(vals)) if len(vals) else float("nan")
        sigma = float(np.std(vals)) if len(vals) else float("nan")

        if len(vals) >= 2:
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                density = kde(x_grid)
                ax.fill_between(x_grid, density, alpha=0.35, color=color, linewidth=0)
                ax.plot(x_grid, density, color=color, linewidth=1.5)
            except np.linalg.LinAlgError:
                pass

        jitter = np.random.default_rng(s).uniform(-0.015, 0.015, size=len(vals))
        ax.plot(vals, jitter, "|", color=color, alpha=0.5, markersize=6, markeredgewidth=1.1)

        ax.set_title(f"step {s}", fontsize=10.5, fontweight="bold", color=color, pad=8)
        ax.text(
            0.97,
            0.95,
            f"n={len(vals)}\nμ={mu:.3f}\nσ={sigma:.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.6),
        )

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(
            "P(score > 0.5) per behaviour" if success_prob else "Mean Judge Score (p_harmful)",
            fontsize=9,
        )
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2, linewidth=0.6, linestyle="--")
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Density across behaviours", fontsize=9.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save:
        out = f"{title.replace(' ', '_')}_behaviour_score_distributions.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {out}")

    plt.show()


def plot_row_score_distributions(
    all_scores,
    row_id,
    steps=None,
    n_distributions=8,
    title=None,
):
    """Distribution of samples at a handful of steps for one behaviour (or
    the aggregate, nanmean'd over behaviours), in the exact per-panel visual
    style of `plot_score_distributions` (gaussian KDE + filled density +
    jittered strip + mu/sigma box) -- but as a single row of panels, one per
    selected step, left to right, without the "average"/"pooled"/"single"
    columns (row_id already picks exactly one pool to show per step).

    all_scores : (n_behaviours, n_runs, n_samples, num_steps) array, e.g.
    the raw all_scores.npy loaded in the forecast-results notebooks.

    row_id : an integer (or numeric string) behaviour index -- pools that
    behaviour's own runs and samples together, n_runs * n_samples points per
    step -- or "aggregate" to nanmean over the behaviour axis first (same
    convention as `row_id="aggregate"` elsewhere in this codebase, e.g.
    `curve_forecast`'s `_get_row_series`), then pool the runs and samples of
    that mean-behaviour together, still n_runs * n_samples points per step.

    steps : explicit list of step (column) indices to show; default is
    `n_distributions` evenly-spaced steps across the steps that actually
    have data.
    """
    from scipy.stats import gaussian_kde

    all_scores = np.asarray(all_scores)
    if all_scores.ndim != 4:
        raise ValueError(
            "all_scores must be 4D (n_behaviours, n_runs, n_samples, num_steps); "
            f"got ndim={all_scores.ndim}"
        )
    n_runs, n_samples, num_steps = all_scores.shape[1:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_scores = (
            np.nanmean(all_scores, axis=0) if row_id == "aggregate" else all_scores[int(row_id)]
        )
    pool = row_scores.reshape(n_runs * n_samples, num_steps)

    if steps is not None:
        step_indices = list(steps)
    else:
        # Scores are often only logged every few steps (e.g. an eval interval),
        # leaving most step-columns entirely NaN -- picking evenly-spaced steps
        # over the full [0, num_steps-1] range would then mostly land on empty
        # columns, so evenly space over the steps that actually have data instead.
        valid_steps = np.flatnonzero(np.any(~np.isnan(pool), axis=0))
        if len(valid_steps) == 0:
            step_indices = []
        elif n_distributions == 1 or len(valid_steps) == 1:
            step_indices = [int(valid_steps[0])]
        else:
            positions = [
                round(i * (len(valid_steps) - 1) / (n_distributions - 1))
                for i in range(n_distributions)
            ]
            step_indices = [int(valid_steps[p]) for p in positions]

    if not step_indices:
        print("plot_row_score_distributions: no non-NaN data found for this row_id, skipping plot.")
        return

    fig, axes = plt.subplots(
        1, len(step_indices), figsize=(2.3 * len(step_indices), 3.2), sharey=True
    )
    axes = np.atleast_1d(axes)

    x_grid = np.linspace(0, 1, 400)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(step_indices) - 1, 1)) for i in range(len(step_indices))]

    for ax, s, color in zip(axes, step_indices, colors):
        vals = pool[:, s]
        vals = vals[~np.isnan(vals)]
        mu = float(np.mean(vals)) if len(vals) else float("nan")
        var = float(np.var(vals)) if len(vals) else float("nan")

        if len(vals) >= 2:
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                density = kde(x_grid)
                ax.fill_between(x_grid, density, alpha=0.35, color=color)
                ax.plot(x_grid, density, color=color, linewidth=1.2)
            except np.linalg.LinAlgError:
                pass

        jitter = np.random.default_rng(s).uniform(-0.015, 0.015, size=len(vals))
        ax.plot(vals, jitter, "|", color=color, alpha=0.25, markersize=4)

        ax.text(
            0.97,
            0.95,
            f"μ={mu:.3f}\nσ²={var:.4f}",
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, lw=0),
        )

        ax.set_title(f"step {s}", fontsize=9)
        ax.set_xlabel("Strong Reject Score (p_harmful)", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0].set_ylabel("density")
    row_label = "aggregate" if row_id == "aggregate" else f"row {row_id}"
    fig.suptitle(
        (f"{title} — " if title else "")
        + f"{row_label}: score distribution over steps ({pool.shape[0]} points/step)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    plt.show()


"""
analyses for scores array of shape (n_behaviours, n_runs, n_samples, n_steps)
= (40, 24, 32, 332)

scores[b, r, s, t] = 1 if attack succeeded on sample s, else 0.
step_times : 1-D array of length n_steps giving the actual optimization
             step number at each index, e.g. np.array([1,2,...,30,80,...,330]).
             Substitute your real values.
"""

import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.optimize import curve_fit

# ── placeholder: replace with your real step indices ──────────────────────────
# e.g. np.concatenate([np.arange(1,31), np.arange(80,331,50)])
step_times = np.arange(232)

# =============================================================================
# UTILITY
# =============================================================================


def compute_rates(scores, threshold=0.5):
    """
    Parameters
    ----------
    scores    : (B, R, S, T)  raw judge scores, NaN where no sample was taken
    threshold : float         a sample counts as a success if score > threshold

    Returns
    -------
    phat   (B, R, T)  per-(behaviour,run,step) success rate
    k      (B, R, T)  success counts (samples exceeding threshold)
    n      (B, R, T)  total sample counts (non-NaN draws)
    """
    valid = ~np.isnan(scores)  # (B, R, S, T) bool
    n = valid.sum(axis=2).astype(float)  # how many real draws
    n = np.where(n == 0, np.nan, n)  # avoid /0

    successes = np.where(valid, scores > threshold, 0)  # threshold before summing
    k = successes.sum(axis=2)  # count above threshold
    phat = k / n
    return phat, k.astype(int), n.astype(int)


# =============================================================================
# STEP 2  Variance decomposition at a fixed late step
# =============================================================================


def step2_variance_decomposition(scores, step_idx=-1):
    """
    Decomposes Var(phat_{b,r}) at a single step into
        σ²_behaviour  : between-behaviour
        σ²_run_true   : between-run within behaviour (sampling-noise corrected)
        noise_floor   : binomial sampling noise
    and reports the ICC.
    """
    phat, k, n = compute_rates(scores)  # (B, R, T)
    phat_t = phat[:, :, step_idx]  # (B, R)
    k_t = k[:, :, step_idx]
    n_t = n[:, :, step_idx]

    B, R = phat_t.shape
    N = B * R
    grand = np.nanmean(phat_t)

    beh_means = np.nanmean(phat_t, axis=1)  # (B,)
    n_i = np.sum(~np.isnan(phat_t), axis=1)  # runs per behaviour

    ss_between = np.nansum(n_i * (beh_means - grand) ** 2)
    ss_within = np.nansum((phat_t - beh_means[:, None]) ** 2)

    ms_between = ss_between / (B - 1)
    ms_within = ss_within / (N - B)

    n0 = (N - np.sum(n_i**2) / N) / (B - 1)  # ~R when balanced

    var_behaviour = max((ms_between - ms_within) / n0, 0.0)
    var_within_raw = ms_within

    # binomial noise floor: E[Var(phat | p)] = p(1-p)/n per cell
    noise_floor = np.nanmean(phat_t * (1 - phat_t) / (n_t - 1))
    var_run_true = max(var_within_raw - noise_floor, 0.0)

    icc_raw = var_behaviour / (var_behaviour + var_within_raw)
    icc_corr = (
        var_behaviour / (var_behaviour + var_run_true)
        if (var_behaviour + var_run_true) > 0
        else np.nan
    )
    noise_frac = 1 - var_run_true / var_within_raw if var_within_raw > 0 else np.nan

    print("=== Step 2: Variance decomposition ===")
    print(f"  step index                       : {step_idx} (step {step_times[step_idx]})")
    print(f"  between-behaviour variance σ²_b  : {var_behaviour:.5f}")
    print(f"  within-behaviour variance (raw)  : {var_within_raw:.5f}")
    print(f"    ↳ binomial noise floor         : {noise_floor:.5f}  ({noise_frac:.1%} of raw)")
    print(f"    ↳ true between-run variance    : {var_run_true:.5f}")
    print(f"  ICC (corrected)                  : {icc_corr:.3f}")
    print()
    print("  Interpretation:")
    if icc_corr > 0.8:
        print("  → Behaviour dominates. Pool runs freely; work with 40 behaviour curves.")
    elif icc_corr > 0.4:
        print("  → Both sources matter. Pool but carry seed CI bands.")
    else:
        print("  → Seed noise dominates. ASR depends heavily on initialisation.")
    print()

    # optional GLMM (requires pymc)
    try:
        _glmm_decomposition(k_t, n_t)
    except ImportError:
        print("  (install pymc for the GLMM version)\n")

    return dict(
        var_behaviour=var_behaviour,
        var_run_true=var_run_true,
        noise_floor=noise_floor,
        icc=icc_corr,
    )


def _glmm_decomposition(k_t, n_t):
    """Binomial logit-normal GLMM — sampling noise is absorbed by likelihood."""
    import arviz as az
    import pymc as pm

    B, R = k_t.shape
    beh_idx = np.repeat(np.arange(B), R)
    k_flat = k_t.ravel()
    n_flat = n_t.ravel()
    mask = ~np.isnan(k_flat)

    with pm.Model():
        mu = pm.Normal("mu", 0, 1.5)
        sigma_beh = pm.HalfNormal("sigma_behaviour", 1.5)
        sigma_run = pm.HalfNormal("sigma_run", 1.5)
        u_raw = pm.Normal("u_raw", 0, 1, shape=B)
        v_raw = pm.Normal("v_raw", 0, 1, shape=mask.sum())
        logit_p = mu + u_raw[beh_idx[mask]] * sigma_beh + v_raw * sigma_run
        pm.Binomial(
            "y", n=n_flat[mask].astype(int), logit_p=logit_p, observed=k_flat[mask].astype(int)
        )
        icc = pm.Deterministic("ICC", sigma_beh**2 / (sigma_beh**2 + sigma_run**2))
        idata = pm.sample(500, tune=500, chains=2, progressbar=False)

    print("\n  === GLMM (logit scale) ===")
    print(az.summary(idata, var_names=["sigma_behaviour", "sigma_run", "ICC"], hdi_prob=0.9))


# =============================================================================
# STEP 3  Per-behaviour trajectory characterisation
# =============================================================================


def step3_behaviour_trajectories(scores, p_thresh=0.5, fig_path=None):
    """
    Pools over runs and samples → p_b(t), shape (B, T).
    Classifies each behaviour and extracts compute-to-break.
    """
    phat, _, _ = compute_rates(scores)  # (B, R, T)
    p_b = np.nanmean(phat, axis=1)  # (B, T) — pool over runs

    B, T = p_b.shape
    final_p = p_b[:, -1]
    initial_p = p_b[:, 0]

    # compute-to-break: first step where p_b(t) > p_thresh
    ctb_idx = np.full(B, np.nan)
    for b in range(B):
        hits = np.where(p_b[b] > p_thresh)[0]
        if len(hits):
            ctb_idx[b] = hits[0]
    ctb_step = np.full(B, np.nan)
    valid = ~np.isnan(ctb_idx)
    ctb_step[valid] = step_times[ctb_idx[valid].astype(int)]

    # classify
    labels = []
    for b in range(B):
        if initial_p[b] > p_thresh:
            labels.append("broken-at-0")
        elif not np.isnan(ctb_idx[b]) and ctb_idx[b] < 30:
            labels.append("early-break")
        elif not np.isnan(ctb_idx[b]):
            labels.append("late-break")
        else:
            labels.append("hard")

    print("=== Step 3: Behaviour trajectory classification ===")
    for cls in ["broken-at-0", "early-break", "late-break", "hard"]:
        n = labels.count(cls)
        print(f"  {cls:<14}: {n:>3} behaviours")
    print(f"\n  Compute-to-break (steps, median)   : {np.nanmedian(ctb_step):.0f}")
    print(
        f"  Compute-to-break (steps, 75th pct) : {np.nanpercentile(ctb_step[~np.isnan(ctb_step)], 75):.0f}"
    )
    print()

    # ── plot ──────────────────────────────────────────────────────────────────
    cmap = {"broken-at-0": "C0", "early-break": "C2", "late-break": "C1", "hard": "C3"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for b in range(B):
        ax.plot(step_times, p_b[b], color=cmap[labels[b]], alpha=0.5, lw=1)
    ax.axhline(p_thresh, ls="--", color="k", lw=0.8)
    for cls, col in cmap.items():
        ax.plot([], [], color=col, label=cls)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("pooled success rate")
    ax.set_title("Per-behaviour trajectories")
    ax.legend(fontsize=8)

    ax = axes[1]
    finite_ctb = ctb_step[~np.isnan(ctb_step)]
    ax.hist(finite_ctb, bins=15, color="steelblue", edgecolor="white")
    ax.set_xlabel("step of first break")
    ax.set_ylabel("count")
    ax.set_title(f"Compute-to-break (n={len(finite_ctb)}, {labels.count('hard')} never break)")

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=150)
    plt.show()

    return p_b, labels, ctb_step


# =============================================================================
# STEP 4  Held-out extrapolation
# =============================================================================


def _sat_exp(t, S_inf, lam):
    """Saturating exponential: S(t) = S_inf * (1 - exp(-lam * t))"""
    return S_inf * (1.0 - np.exp(-lam * np.maximum(t, 0)))


def step4_extrapolation(scores, cutoffs_idx=None, target_idx=-1, fig_path=None):
    """
    For each cutoff in cutoffs_idx (list of step indices), fits a saturating
    exponential to data up to that cutoff and predicts the value at target_idx.
    Reports mean absolute error across behaviours.

    Does this TWICE:
        (a) on the aggregate mean curve
        (b) per-behaviour, then averages predictions → tests our main claim
    """
    phat, _, _ = compute_rates(scores)  # (B, R, T)
    p_b = np.nanmean(phat, axis=1)  # (B, T)
    S_agg = p_b.mean(axis=0)  # (T,) aggregate

    if cutoffs_idx is None:
        # sensible defaults: 10%, 20%, 30%, 50%, 70% of timeline
        n_t = scores.shape[3]
        cutoffs_idx = [int(f * n_t) for f in [0.1, 0.2, 0.3, 0.5, 0.7]]
        cutoffs_idx = [max(c, 5) for c in cutoffs_idx]  # need ≥5 pts to fit

    target_step = step_times[target_idx]
    true_agg = float(S_agg[target_idx])
    true_per_b = p_b[:, target_idx]

    rows = []
    for cidx in cutoffs_idx:
        t_fit = step_times[:cidx].astype(float)
        t_pred = np.array([float(target_step)])

        # ── (a) aggregate fit ────────────────────────────────────────────────
        y_fit = S_agg[:cidx]
        try:
            (S_inf, lam), _ = curve_fit(
                _sat_exp, t_fit, y_fit, p0=[0.8, 0.01], bounds=([0, 1e-6], [1, 10]), maxfev=5000
            )
            pred_agg = float(_sat_exp(t_pred, S_inf, lam))
        except RuntimeError:
            pred_agg = np.nan

        # ── (b) per-behaviour fits, then average ─────────────────────────────
        preds_b = []
        for b in range(p_b.shape[0]):
            y_b = p_b[b, :cidx]
            if np.all(y_b == 0):
                preds_b.append(0.0)
                continue
            try:
                (S_inf_b, lam_b), _ = curve_fit(
                    _sat_exp,
                    t_fit,
                    y_b,
                    p0=[np.max(y_b) + 0.05, 0.01],
                    bounds=([0, 1e-6], [1, 10]),
                    maxfev=5000,
                )
                preds_b.append(float(_sat_exp(t_pred, S_inf_b, lam_b)))
            except RuntimeError:
                preds_b.append(np.nan)

        pred_per_b_agg = float(np.nanmean(preds_b))
        mae_per_b = float(np.nanmean(np.abs(np.array(preds_b) - true_per_b)))

        rows.append(
            dict(
                cutoff_step=step_times[cidx],
                obs_fraction=cidx / scores.shape[3],
                pred_agg_direct=pred_agg,
                pred_agg_via_b=pred_per_b_agg,
                true_agg=true_agg,
                err_direct=abs(pred_agg - true_agg) if not np.isnan(pred_agg) else np.nan,
                err_via_b=abs(pred_per_b_agg - true_agg),
                mae_per_b=mae_per_b,
            )
        )

    df = pd.DataFrame(rows)
    print("=== Step 4: Held-out extrapolation ===")
    print(f"  target step : {target_step}  (true aggregate ASR = {true_agg:.3f})")
    print()
    print(
        df[
            [
                "cutoff_step",
                "obs_fraction",
                "pred_agg_direct",
                "pred_agg_via_b",
                "err_direct",
                "err_via_b",
            ]
        ].to_string(index=False)
    )
    print()

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(df["obs_fraction"], df["err_direct"], "o-", label="direct aggregate fit")
    ax.plot(df["obs_fraction"], df["err_via_b"], "s-", label="per-behaviour then average")
    ax.axhline(0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("fraction of budget observed")
    ax.set_ylabel("|error| in ASR")
    ax.set_title("Extrapolation error vs observed budget")
    ax.legend()

    ax = axes[1]
    ax.plot(step_times, S_agg, "k-", label="true aggregate")
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cutoffs_idx)))
    for i, row in df.iterrows():
        if not np.isnan(row["pred_agg_direct"]):
            ax.scatter(
                target_step,
                row["pred_agg_direct"],
                color=colors[i],
                zorder=3,
                label=f"step {int(row['cutoff_step'])}",
            )
    ax.set_xlabel("step")
    ax.set_ylabel("aggregate ASR")
    ax.set_title("Predicted vs true final ASR")
    ax.legend(fontsize=7)

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=150)
    plt.show()

    return df


# =============================================================================
# STEP 5  Edge / crossover probe
# =============================================================================


def step5_edge_crossover(scores, n_edge=5, alpha=0.05, fig_path=None):
    """
    Pools ALL runs and samples (up to ~1024 draws per behaviour per step).
    For each step:
      • Records the per-behaviour pooled rates
      • Tracks p_min(t)  — the safest behaviour's rate  (proxy for hat_p)
      • Detects when p_min lifts off zero  (hat_mu onset)
      • Tracks the distribution of rates over behaviours (histogram animation
        replaced here by a summary heatmap)
    """
    B, R, x, T = scores.shape
    # pool over runs and samples → (B, T)
    k_pool = np.nansum(scores.reshape(B, R * x, T), axis=1)
    n_pool = (~np.isnan(scores.reshape(B, R * x, T))).sum(axis=1).astype(float)
    n_pool = np.where(n_pool == 0, np.nan, n_pool)
    p_pool = k_pool / n_pool  # (B, T)

    # ── lift-off detection ────────────────────────────────────────────────────
    # p_min(t): for each step, min pooled rate across behaviours
    p_min = np.nanmin(p_pool, axis=0)  # (T,)

    # Binomial p-value for observing k>=1 successes under null p=0
    # (exact: P(X>=1 | n, p=0) only = 1 iff k>0)
    # We use a one-sided binomial CI lower bound instead:
    # if the LOWER end of the Wilson CI > 0, we call lift-off
    n_safe = n_pool[np.argmin(p_pool, axis=0), np.arange(T)]
    k_safe = k_pool[np.argmin(p_pool, axis=0), np.arange(T)]
    # Wilson lower bound for the "safest" behaviour at each step
    z = 1.96
    n_w = np.where(n_safe > 0, n_safe, np.nan)
    p_w = k_safe / n_w
    wilson_lo = (
        p_w + z**2 / (2 * n_w) - z * np.sqrt(p_w * (1 - p_w) / n_w + z**2 / (4 * n_w**2))
    ) / (1 + z**2 / n_w)
    liftoff_idx = np.where(wilson_lo > 0)[0]
    liftoff_step = int(step_times[liftoff_idx[0]]) if len(liftoff_idx) else None

    print("=== Step 5: Edge / crossover probe ===")
    print(f"  total draws per (behaviour, step)   : ~{int(np.nanmedian(n_pool))}")
    print(f"  p_min at step 0                     : {p_min[0]:.4f}")
    print(f"  p_min at final step                 : {p_min[-1]:.4f}")
    if liftoff_step is not None:
        print(f"  hat_mu onset (Wilson CI lifts off 0): step {liftoff_step}")
        print(f"  → crossover expected AFTER step {liftoff_step}")
    else:
        print("  hat_mu = 0 throughout: safest behaviour never reliably breaks")
        print("  → polynomial ASR scaling regime, no crossover detected")
    print()

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 3])

    # (a) trajectory of edge quantiles
    ax0 = fig.add_subplot(gs[0])
    for q, col in [(0, "C3"), (0.1, "C1"), (0.5, "C2"), (0.9, "C0")]:
        ax0.plot(step_times, np.nanquantile(p_pool, q, axis=0), color=col, label=f"q={q:.0%}")
    if liftoff_step is not None:
        ax0.axvline(liftoff_step, ls="--", color="k", lw=0.8, label="lift-off")
    ax0.set_xlabel("step")
    ax0.set_ylabel("pooled success rate")
    ax0.set_title("Distribution quantiles over behaviours")
    ax0.legend(fontsize=7)

    # (b) p_min and Wilson CI
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(step_times, p_min, "C3-", label="p_min (observed)")
    ax1.fill_between(
        step_times, 0, wilson_lo, alpha=0.3, color="C3", label="Wilson lower CI > 0 → lift-off"
    )
    if liftoff_step is not None:
        ax1.axvline(liftoff_step, ls="--", color="k", lw=0.8)
    ax1.set_xlabel("step")
    ax1.set_ylabel("rate of safest behaviour")
    ax1.set_title("hat_p / hat_mu onset")
    ax1.legend(fontsize=7)

    # (c) behaviour × step heatmap
    ax2 = fig.add_subplot(gs[2])
    sort_idx = np.argsort(p_pool[:, -1])[::-1]
    im = ax2.imshow(
        p_pool[sort_idx],
        aspect="auto",
        origin="upper",
        extent=[step_times[0], step_times[-1], B, 0],
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )
    ax2.set_xlabel("step")
    ax2.set_ylabel("behaviour (sorted by final rate)")
    ax2.set_title("Pooled success rate heatmap")
    plt.colorbar(im, ax=ax2, label="success rate")

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=150)
    plt.show()

    return dict(p_pool=p_pool, p_min=p_min, wilson_lo=wilson_lo, liftoff_step=liftoff_step)
