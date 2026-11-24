"""Shared helpers for the per-attack forecast notebooks (gcg_forecast.ipynb and friends).

Keeps the logic that's identical across every attack/model notebook in one place, so the
notebooks themselves stay thin: config (run name, behaviour list, categories, cutoffs) plus
calls into this module and into thesis.analysis.forecast/plot_forecasts/forecast_success_probs,
with whatever attack-specific exploration cells on top.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from thesis.analysis.analyse import step2_variance_decomposition
from thesis.analysis.forecast import fit_and_forecast
from thesis.analysis.forecast_success_probs import build_asr_probs_from_fit, fit_success_probs
from thesis.analysis.plot_forecasts import asr_at_k_forecast_error, build_asr_probs

# The 40-behaviour GuidedBench subset's category labels, shared across every attack/model run
# that uses this behaviour-ID numbering -- check a run's own behaviour_order.json for which
# subset of these IDs it actually covers (load_forecast_run skips any category member a run
# doesn't have).
CATEGORIES_BY_NUMBER = {
    "misinformation": [170, 171, 172, 173, 174, 175, 121, 122, 115, 21],
    "offensive": [111, 112, 116, 123, 84, 86, 90, 132, 141, 149],
    "malicious_creation": [37, 42, 44, 49, 45, 50, 58, 59, 61, 46],
    "harmful_instruction": [85, 77, 76, 70, 66, 53, 43, 41, 60, 27],
}


def load_forecast_run(forecast_run, list_passed_in_run=None, categories_by_number=None,
                       scores_dir="../data/scores", threshold=0.5):
    """Load `<scores_dir>/<forecast_run>/all_scores.npy` and derive the harmfulness/success/ASR
    arrays every notebook starts from.

    list_passed_in_run : behaviour IDs (axis-0 order of all_scores.npy) for this run. If None,
        read from `<scores_dir>/<forecast_run>/behaviour_order.json` (present for runs that ship
        one; older runs need it passed explicitly).
    categories_by_number : {category: [behaviour IDs]}, mapped to axis-0 indices via
        list_passed_in_run. IDs not present in this run (e.g. a category dict shared across runs
        with different behaviour subsets) are silently skipped. None/omitted -> no categories.

    Returns a dict with all_scores, categories_to_idx, avg_harmfulness, var_harmfulness,
    success, success_prob, var_success, observed_asr, independent_asr, constant_hazard_asr.
    """
    scores_cache_dir = f"{scores_dir}/{forecast_run}"
    all_scores = np.load(f"{scores_cache_dir}/all_scores.npy")

    if list_passed_in_run is None:
        order_path = f"{scores_cache_dir}/behaviour_order.json"
        try:
            with open(order_path) as f:
                list_passed_in_run = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"{order_path} not found -- pass list_passed_in_run explicitly for this run."
            )

    mapping = {num: idx for idx, num in enumerate(list_passed_in_run)}
    categories_to_idx = {}
    for category, numbers in (categories_by_number or {}).items():
        idxs = [mapping[num] for num in numbers if num in mapping]
        skipped = len(numbers) - len(idxs)
        if skipped:
            print(f"  {forecast_run}: {skipped} '{category}' behaviour(s) not in this run, skipped")
        categories_to_idx[category] = idxs

    # Calculate avg. harmfulness, hazard & ASR trajectories
    avg_harmfulness = np.nanmean(all_scores, axis=(1, 2))
    var_harmfulness = np.nanvar(all_scores, axis=(1, 2))
    success = np.where(np.isnan(all_scores), np.nan, (all_scores > threshold).astype(float))

    # Expected value of probability that sample i is larger than the threshold
    success_prob = np.nanmean(success, axis=(1, 2))          # (n_behaviors, n_steps)
    var_success = np.nanvar(success_prob, axis=(0))

    # Observed ASR
    any_sample_above = np.nanmax(success, axis=2)   # max over samples (axis=2) -> (problems, runs, steps)
    ever_valid = np.maximum.accumulate(~np.isnan(any_sample_above), axis=-1)
    cum_success = np.maximum.accumulate(np.nan_to_num(any_sample_above), axis=-1)
    observed_asr = np.nanmean(np.where(ever_valid, cum_success, np.nan), axis=1)  # (problems, steps)

    # Independent ASR
    per_step = np.nanmean(any_sample_above, axis=1)  # (problems, steps): P(any sample succeeds) at this step
    per_step_ever_valid = np.maximum.accumulate(~np.isnan(per_step), axis=1)
    independent_asr = np.where(per_step_ever_valid, 1 - np.nancumprod(1 - per_step, axis=1), np.nan)

    # Independent + Constant Success Prob ASR
    n_ref = max(1, round(0.2 * per_step.shape[1]))
    per_step_ref = np.nanmean(per_step[:, :n_ref], axis=1, keepdims=True)  # (problems, 1)
    step_valid = ~np.isnan(per_step)  # the real trajectory's own recording schedule
    per_step_const_full = np.where(step_valid, np.broadcast_to(per_step_ref, per_step.shape), np.nan)
    ref_valid = np.broadcast_to(~np.isnan(per_step_ref), per_step.shape)  # was the reference window itself defined?
    constant_hazard_asr = np.where(
        per_step_ever_valid & ref_valid, 1 - np.nancumprod(1 - per_step_const_full, axis=1), np.nan
    )

    print(all_scores.shape)
    print(success.shape)
    print(success_prob.shape)

    return {
        "all_scores": all_scores,
        "categories_to_idx": categories_to_idx,
        "avg_harmfulness": avg_harmfulness,
        "var_harmfulness": var_harmfulness,
        "success": success,
        "success_prob": success_prob,
        "var_success": var_success,
        "observed_asr": observed_asr,
        "independent_asr": independent_asr,
        "constant_hazard_asr": constant_hazard_asr,
    }


def summarize_fit_metrics(df, families=None, metrics=(), sort_by="in_sample_rmse", by_behaviour=True, ascending=True):
    """Compare `metrics` (plus `in_sample_rmse`) across family x cutoff.

    Grouped by cutoff, with families ranked best-to-worst (ascending `sort_by` by default)
    within each group, so the best-fitting family per cutoff is immediately visible at the
    top of each block.

    If `by_behaviour` (default), aggregate over individual behaviour rows (excluding
    `aggregate`): mean and median per (family, cutoff). If False, use the `aggregate` row
    directly: one row per (family, cutoff) with the raw metric values.
    """
    sub = df[df["row_id"] != "aggregate"] if by_behaviour else df[df["row_id"] == "aggregate"]
    if families is not None:
        sub = sub[sub["family"].isin(families)]
    cols = ["in_sample_rmse", *metrics]

    if by_behaviour:
        summary = sub.groupby(["cutoff", "family"])[cols].agg(["mean", "median"])
        sort_key = (sort_by, "median")
    else:
        summary = sub.set_index(["cutoff", "family"])[cols]
        sort_key = sort_by

    summary = summary.sort_values(sort_key, ascending=ascending)
    return summary.sort_index(level="cutoff", sort_remaining=False, kind="stable")


def fit_success_prob_and_build_probs(
    name,
    success_prob,
    all_scores,
    cutoffs,
    family="power_law_saturating",
    fit_cutoff_for_probs=None,
    k_values=(1, 100),
    threshold=0.5,
):
    """Fit `family` to a success-probability trajectory across `cutoffs`
    (`forecast.fit_and_forecast`), print per-cutoff convergence and ASR-forecast-error
    diagnostics (`plot_forecasts.build_asr_probs` + `asr_at_k_forecast_error`), and build the
    forecast-vs-observed table at `fit_cutoff_for_probs` (defaults to the last entry of
    `cutoffs`) for plotting (e.g. `plot_asr_compute_heatmap`/`plot_asr_decomposition`).

    Returns (fit_results, probs).
    """
    cutoffs = list(cutoffs)
    if fit_cutoff_for_probs is None:
        fit_cutoff_for_probs = cutoffs[-1]

    fit_results = fit_and_forecast(success_prob, cutoffs=cutoffs, families=[family])

    totals = fit_results.groupby(["family", "cutoff"]).size()
    failed = (~fit_results["converged"]).groupby(
        [fit_results["family"], fit_results["cutoff"]]
    ).sum()
    print(f"{name}:")
    for (fam, cutoff), n_failed in failed.items():
        print(f"  {fam} @ cutoff={cutoff}: {n_failed}/{totals[(fam, cutoff)]} did not converge")

    for cutoff in cutoffs:
        probs_c = build_asr_probs(fit_results, all_scores, family=family, fit_cutoff=cutoff, threshold=threshold)
        for k in k_values:
            rmse_asr, endpoint_err_asr = asr_at_k_forecast_error(probs_c, k=k, cutoff=cutoff, hypergeometric=True)
            print(
                f"  cutoff={cutoff}, k={k}: ASR RMSE (behaviour-averaged) = {rmse_asr:.4f}, "
                f"holdout endpoint error = {endpoint_err_asr}"
            )

    probs = build_asr_probs(
        fit_results, all_scores, family=family, fit_cutoff=fit_cutoff_for_probs, threshold=threshold
    )
    return fit_results, probs


def investigate_non_convergence(df, data, failure_reason, family="power_law_saturating"):
    """Diagnose why rows failed with `failure_reason` for `family`: per failing row, checks
    whether the observed trajectory is just flat (no rise) or lacks detectable deceleration.
    `df`/`data` are a `fit_and_forecast` result and the trajectory it was fit to (e.g.
    `success_prob`). Prints a count of failures by `likely_cause`; use it after the convergence
    printout in `fit_success_prob_and_build_probs` flags something worth a closer look.
    """
    import pandas as pd

    fails = df[
        (df["family"] == family)
        & (~df["converged"])
        & (df["failure_reason"] == failure_reason)
    ].copy()
    if fails.empty:
        print(f"No rows match family={family!r} failure_reason={failure_reason!r}.")
        return

    def row_array(row_id):
        return np.nanmean(data, axis=0) if row_id == "aggregate" else data[int(row_id)]

    def row_diagnostics(row_id):
        y = row_array(row_id)
        t_all = np.arange(len(y))
        mask = ~np.isnan(y)
        t, valid = t_all[mask], y[mask]
        if len(valid) < 5:
            return pd.Series({
                "obs_max": float(valid.max()) if len(valid) else np.nan,
                "frac_zero": float(np.mean(valid == 0)) if len(valid) else np.nan,
                "rate_pvalue_one_sided": np.nan,
            })
        dt = np.diff(t)
        rate = np.diff(valid) / dt
        mid = t[:-1] + dt / 2.0
        lin = stats.linregress(mid, rate)
        p_one_sided = lin.pvalue / 2.0 if lin.slope < 0 else 1.0  # matches forecast.py's one-sided convention
        return pd.Series({
            "obs_max": float(valid.max()),
            "frac_zero": float(np.mean(valid == 0)),
            "rate_pvalue_one_sided": p_one_sided,
        })

    diag = fails["row_id"].apply(row_diagnostics)
    fails = pd.concat(
        [fails[["row_id", "cutoff", "n_fit_points", "failure_detail", "curvature_detected"]].reset_index(drop=True),
         diag.reset_index(drop=True)],
        axis=1,
    )

    def classify(r):
        if r["obs_max"] < 0.02:
            return "flat (nonzero level, no rise)"
        if not r["curvature_detected"]:
            return "no significant deceleration (curvature test found nothing)"
        return "other / inspect manually"

    fails["likely_cause"] = fails.apply(classify, axis=1)
    return fails.sort_values(["cutoff", "row_id"]).value_counts(["row_id"])


def fit_success_probs_bayesian_report(
    name,
    pooled_success,
    categories_to_idx,
    all_scores,
    cutoffs=(200, 400, 600, 800, "max"),
    step_multiplier=1,
    num_warmup=2000,
    num_samples=2000,
    num_chains=1,
    threshold=0.5,
    q=0.5,
    k_values=(1, 100),
):
    """Hierarchical-Bayes counterpart of `fit_success_prob_and_build_probs`:
    `forecast_success_probs.fit_success_probs` per cutoff, each posterior turned into an ASR
    forecast via `build_asr_probs_from_fit`, with the same RMSE/endpoint-error diagnostics
    printed per cutoff.

    `step_multiplier` scales the step index into the model's own time covariate (GCG's original
    notebooks used 10 for historical reasons; defaults to 1, i.e. a plain step count).

    Returns {cutoff: (fit, probs_bayes)}, in `cutoffs` order. This is the slow path (MCMC per
    cutoff) -- shrink `num_warmup`/`num_samples` for a quick smoke test.
    """
    cutoffs = list(cutoffs)
    n_steps = pooled_success.shape[-1]
    t_values = step_multiplier * (np.arange(n_steps) + 1)

    fits = fit_success_probs(
        pooled_success, categories_to_idx, cutoffs=cutoffs, t_values=t_values,
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
    )

    results = {}
    print(f"{name}:")
    for cutoff, fit in fits.items():
        probs_bayes, rmse_bayes, endpoint_err_bayes = build_asr_probs_from_fit(
            fit, all_scores, threshold=threshold, q=q,
        )
        for k in k_values:
            rmse_asr, endpoint_err_asr = asr_at_k_forecast_error(
                probs_bayes, k=k, cutoff=fit.cutoff, hypergeometric=True
            )
            print(
                f"  cutoff={cutoff}, k={k}: p_forecast RMSE={rmse_bayes:.4f} (endpoint {endpoint_err_bayes}); "
                f"ASR RMSE={rmse_asr:.4f} (endpoint {endpoint_err_asr})"
            )
        results[cutoff] = (fit, probs_bayes)
    return results


def print_variance_decomposition(models):
    """Per-model judge-dispersion/run-variance summary: for each `(name, all_scores, bayes_fit)`
    in `models`, prints the Bayesian fit's phi (judge dispersion) median -- pass `bayes_fit=None`
    to skip -- and `analyse.step2_variance_decomposition`'s `var_run_true` for `all_scores`.
    """
    for name, all_scores, bayes_fit in models:
        if bayes_fit is not None:
            print(name, "phi median:", np.median(1.0 / bayes_fit.concentration))
        result = step2_variance_decomposition(all_scores)
        print(name, "var_run_true:", result["var_run_true"])
