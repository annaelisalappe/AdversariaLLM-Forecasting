"""
Build the `data/forecasts/<run>/{judge_score_per_behaviour,asr_per_behaviour}/` layout from a
run's cached `data/scores/<run>/all_scores.npy`.

This format was originally produced on a cluster by a script that is no longer present in this
repo -- the existing `data/forecasts/gcg_llama_2400/` files were downloaded manually, not
generated here. This module is a reconstruction of that format from the column names/shapes of
that existing output (all of which turn out to be plain `fit_and_forecast` output columns, see
`forecast.py`), cross-checked against `gcg_llama_2400`'s own `summary.csv`/`full_data_summary.csv`
-- not a verified byte-for-byte match, since the two pickles need `pyarrow` to read in this
environment and a bootstrap `k`/cutoff list for that specific run is not recoverable from the
CSVs alone.

Per run, for each of two per-behaviour trajectories (the deterministic judge score, and the
bootstrapped ASR@k), `fit_and_forecast` is called once across `cutoffs + ["max"]`, and the
result is split into:
    results.pkl           raw `fit_and_forecast` output (every row, every cutoff)
    summary.csv           per-family rollup over the non-"max" rows (pooled convergence /
                           holdout / curvature stats, plus one `aggregate_cutoff{c}_*` column
                           block per numeric cutoff)
    full_data_fits.pkl     the `cutoff == "max"` subset of `results.pkl`
    full_data_summary.csv  per-family rollup of that full-data (in-sample only) subset
`asr_per_behaviour/` additionally gets `boot_std.npy` / `agg_boot_std.npy` -- the per-behaviour
and aggregate standard deviation across `compute_asr_at_k_cumulative`'s bootstrap replicates.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_scores(scores_dir: str, run: str) -> np.ndarray:
    """Load `<scores_dir>/<run>/all_scores.npy` (the `save_scores` cache)."""
    return np.load(os.path.join(scores_dir, run, "all_scores.npy"))


def summarize_multi_cutoff(
    results: pd.DataFrame, families: Sequence[str], numeric_cutoffs: Sequence[int]
) -> pd.DataFrame:
    """Per-family rollup of a multi-cutoff `fit_and_forecast` result, matching the recovered
    `summary.csv` layout: pooled convergence/holdout/curvature stats across every (row,
    numeric cutoff) pair -- behaviour rows and the aggregate row alike -- plus one
    `aggregate_cutoff{c}_*` column block per cutoff, taken from that cutoff's own aggregate row.
    Rows with cutoff == "max" are excluded (they carry no holdout/asymptote-error fields).
    """
    pool = results[results["cutoff"] != "max"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN median, e.g. no asymptote param
        for family in families:
            g = pool[pool["family"] == family]
            non_converged = g.loc[~g["converged"], "failure_reason"]
            if len(non_converged):
                counts = non_converged.value_counts()
                top_reason, top_count = counts.index[0], int(counts.iloc[0])
            else:
                top_reason, top_count = None, 0

            row = {
                "family": family,
                "n_attempted": len(g),
                "n_converged": int(g["converged"].sum()),
                "convergence_rate": g["converged"].mean() if len(g) else np.nan,
                "top_failure_reason": top_reason,
                "top_failure_reason_count": top_count,
                "median_holdout_mae_normalized": g["holdout_mae_normalized"].median(),
                "median_asymptote_rel_error": g["asymptote_rel_error"].median(),
                "n_rows_with_curvature_detected": int(g["curvature_detected"].sum()),
                "n_rows_total": len(g),
            }
            for c in numeric_cutoffs:
                at_c = g[(g["row_id"] == "aggregate") & (g["cutoff"] == c)]
                r = at_c.iloc[0] if len(at_c) else None
                prefix = f"aggregate_cutoff{c}_"
                row[prefix + "converged"] = r["converged"] if r is not None else None
                row[prefix + "failure_reason"] = r["failure_reason"] if r is not None else None
                row[prefix + "holdout_mae_normalized"] = (
                    r["holdout_mae_normalized"] if r is not None else None
                )
                row[prefix + "asymptote_rel_error"] = (
                    r["asymptote_rel_error"] if r is not None else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_full_data(results: pd.DataFrame, families: Sequence[str]) -> pd.DataFrame:
    """Per-family rollup of the `cutoff == "max"` subset of a `fit_and_forecast` result,
    matching the recovered `full_data_summary.csv` layout: full-data (in-sample only, no
    holdout) stats for the individual behaviour rows, plus the single aggregate row's own
    values.
    """
    max_only = results[results["cutoff"] == "max"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN mean/median if all non-converged
        for family in families:
            g = max_only[max_only["family"] == family]
            behaviours = g[g["row_id"] != "aggregate"]
            agg = g[g["row_id"] == "aggregate"]
            agg_row = agg.iloc[0] if len(agg) else None
            rows.append(
                {
                    "family": family,
                    "n_behaviour_rows": len(behaviours),
                    "n_behaviour_converged": int(behaviours["converged"].sum()),
                    "behaviour_convergence_rate": (
                        behaviours["converged"].mean() if len(behaviours) else np.nan
                    ),
                    "mean_behaviour_in_sample_rmse": behaviours["in_sample_rmse"].mean(),
                    "median_behaviour_in_sample_rmse": behaviours["in_sample_rmse"].median(),
                    "mean_behaviour_aic": behaviours["aic"].mean(),
                    "aggregate_converged": agg_row["converged"] if agg_row is not None else None,
                    "aggregate_failure_reason": (
                        agg_row["failure_reason"] if agg_row is not None else None
                    ),
                    "aggregate_in_sample_rmse": (
                        agg_row["in_sample_rmse"] if agg_row is not None else None
                    ),
                    "aggregate_aic": agg_row["aic"] if agg_row is not None else None,
                }
            )
    return pd.DataFrame(rows)


def build_forecast_outputs(
    run: str,
    scores_dir: str,
    out_dir: str,
    k: int,
    cutoffs: Sequence[int],
    threshold: float = 0.5,
    families: Sequence[str] | None = None,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    seed: int = 0,
    n_jobs: int = -1,
) -> Path:
    """Reproduce `<out_dir>/<run>/{judge_score_per_behaviour,asr_per_behaviour}/` from
    `<scores_dir>/<run>/all_scores.npy` -- see module docstring for exactly what each output
    file is and how it's derived.

    k, threshold : passed to the bootstrapped ASR@k trajectory (`compute_asr_at_k_cumulative`
        on `all_scores > threshold`, pooled over runs/samples per behaviour).
    cutoffs : numeric holdout cutoffs to fit at (column indices); `"max"` is always added
        implicitly, since every row's asymptote ground truth and the full-data outputs need it.
    families : defaults to every currently registered family (`forecast.ALL_FAMILY_NAMES`).
        Note: the one archived example this format was recovered from (`gcg_llama_2400`) used 8
        families, 3 of which (`log_linear`, `cure_exponential`, `cure_weibull`) are no longer in
        the registry (they're commented out in `forecast.py`) -- new runs will only get the
        current 5 unless those are re-enabled.

    Returns the `<out_dir>/<run>` path written to.
    """
    from thesis.analysis.forecast import ALL_FAMILY_NAMES, fit_and_forecast
    from thesis.analysis.plot_forecasts import compute_asr_at_k_cumulative

    all_scores = load_all_scores(scores_dir, run)
    families = list(families) if families is not None else list(ALL_FAMILY_NAMES)
    numeric_cutoffs = list(cutoffs)
    all_cutoffs = [*numeric_cutoffs, "max"]

    judge_score = np.nanmean(all_scores, axis=(1, 2))  # (n_behaviours, n_steps)
    success = np.where(np.isnan(all_scores), np.nan, (all_scores > threshold).astype(float))
    pooled_success = success.reshape(success.shape[0], -1, success.shape[-1])
    boot = compute_asr_at_k_cumulative(pooled_success, k=k, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    run_dir = Path(out_dir) / run
    for subdir, trajectory, boot_std in (
        ("judge_score_per_behaviour", judge_score, None),
        ("asr_per_behaviour", boot["per_behaviour"], (boot["std"], boot["agg_std"])),
    ):
        out_sub = run_dir / subdir
        out_sub.mkdir(parents=True, exist_ok=True)

        results = fit_and_forecast(
            trajectory,
            families=families,
            cutoffs=all_cutoffs,
            include_aggregate=True,
            n_jobs=n_jobs,
            random_seed=seed,
        )
        results.to_pickle(out_sub / "results.pkl")
        summarize_multi_cutoff(results, families, numeric_cutoffs).to_csv(
            out_sub / "summary.csv", index=False
        )

        full_data = results[results["cutoff"] == "max"]
        full_data.to_pickle(out_sub / "full_data_fits.pkl")
        summarize_full_data(results, families).to_csv(out_sub / "full_data_summary.csv", index=False)

        if boot_std is not None:
            std, agg_std = boot_std
            np.save(out_sub / "boot_std.npy", std)
            np.save(out_sub / "agg_boot_std.npy", agg_std)

        print(f"Wrote {out_sub}/")

    return run_dir
