"""
Method
------
1. Aggregate the noisy-environment sample axis (mean or median) into one
   fitness value per (problem, run, step).
2. For each run, find its first step with a valid (non-nan) measurement,
   and re-express the rest of that run as the DIFFERENCE from that first
   value: diff(step) = value(step) - value(first_valid_step).
   This removes the confound of "this run just happened to start at a
   good/bad candidate" and isolates genuine improvement over the course
   of the run.
3. Fit a robust (Theil-Sen) slope of diff vs. offset-from-start for each
   run. Theil-Sen (median of pairwise slopes) is used instead of
   ordinary least squares because it is not distorted by a few extreme
   step-to-step values ("spiky" candidates), which OLS would be.
4. Test whether the population of per-run slopes is systematically above
   zero using a SIGN-FLIP (Fisher randomisation) permutation test: flip
   the sign of each run's slope at random, recompute the statistic (the
   median slope by default), many times, and see where the real value
   falls in that null distribution.

   This is deliberately NOT a within-run permutation test (shuffling step
   order within a single run). If candidates are generated with
   autocorrelation -- e.g. each candidate is built from the previous one,
   as in local search / mutation-based methods -- shuffling within a run
   destroys that real structure and produces an artificially narrow null,
   which inflates false positives (empirically, this can reach ~47% at
   alpha=0.05 under a driftless but autocorrelated process). The sign-flip
   test instead treats each independent run/seed as one exchangeable
   unit, which is valid regardless of how strong the within-run
   autocorrelation is -- it only assumes runs are mutually independent
   (true by construction: different seeds) and that under "no genuine
   optimisation" each run's slope is equally likely to be positive or
   negative.

Usage
-----
    result = test_optimization(trajectories)  # trajectories: 4D array
    print(result)

    trajectories shape: (n_problems, n_samples, n_runs, n_steps)
      n_problems - independent benchmark problems, tested separately
      n_samples  - repeated draws from a noisy environment AT a given step
      n_runs     - independent seeds of the algorithm
      n_steps    - optimisation steps
    Missing samples should be np.nan. If every sample at a
    (problem, run, step) is nan, that step is treated as missing.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Step 1: aggregate the noisy-environment sample axis
# --------------------------------------------------------------------------

def _aggregate_samples(trajectories: np.ndarray, agg: str = "mean") -> np.ndarray:
    """(n_problems, n_samples, n_runs, n_steps) -> (n_problems, n_runs, n_steps)"""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if agg == "mean":
            return np.nanmean(trajectories, axis=1)
        elif agg == "median":
            return np.nanmedian(trajectories, axis=1)
        else:
            raise ValueError("agg must be 'mean' or 'median'")


# --------------------------------------------------------------------------
# Step 2-3: per-run robust slope, relative to each run's own first
# valid measurement (removes "started at a good candidate" confound)
# --------------------------------------------------------------------------

def _per_run_slope_from_start(run_series: np.ndarray) -> float:
    """
    run_series: 1D array over steps for a single (problem, run), already
    aggregated over samples, possibly with leading/trailing/interior NaNs.

    Finds the first non-nan step, expresses everything from there on as a
    difference from that first value, and fits a Theil-Sen slope of that
    difference against offset-from-start. Requires >=3 valid points past
    (and including) the first valid step; otherwise returns NaN.
    """
    valid_idx = np.where(~np.isnan(run_series))[0]
    if len(valid_idx) == 0:
        return np.nan

    first = valid_idx[0]
    tail = run_series[first:]                 # from first valid step onward
    baseline = tail[0]                         # value at first valid step
    diffs = tail - baseline                    # diffs[0] == 0 by construction
    offsets = np.arange(len(tail))

    valid_tail = ~np.isnan(diffs)
    if valid_tail.sum() < 3:
        return np.nan

    x = offsets[valid_tail].astype(float)
    y = diffs[valid_tail].astype(float)
    return float(stats.theilslopes(y, x).slope)


def _problem_slopes(problem_traj: np.ndarray) -> np.ndarray:
    """problem_traj: (n_runs, n_steps), sample-aggregated. One slope per run."""
    return np.array([_per_run_slope_from_start(problem_traj[r])
                      for r in range(problem_traj.shape[0])])


# --------------------------------------------------------------------------
# Step 4: sign-flip permutation test across runs
# --------------------------------------------------------------------------

def _sign_flip_test(slopes: np.ndarray, n_perm: int = 9999, stat: str = "median",
                     rng: np.random.Generator | None = None) -> dict:
    """
    slopes: 1D array of per-run slopes for one problem (NaNs dropped).

    Randomly flips the sign of each run's slope, recomputes the summary
    statistic, and repeats n_perm times to build an exact null. Valid
    regardless of within-run autocorrelation -- see module docstring.
    """
    if rng is None:
        rng = np.random.default_rng()

    slopes = slopes[~np.isnan(slopes)]
    n = len(slopes)
    if n < 2:
        return {"observed_slope": np.nan, "p_value": np.nan, "n_runs_used": n}

    f = np.median if stat == "median" else np.mean
    observed = f(slopes)

    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        null[i] = f(slopes * signs)

    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return {"observed_slope": float(observed), "p_value": p_value, "n_runs_used": n}


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Simple BH FDR correction across problems. NaNs pass through as NaN."""
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan)
    valid_idx = np.where(~np.isnan(pvalues))[0]
    m = len(valid_idx)
    if m == 0:
        return out
    order = valid_idx[np.argsort(pvalues[valid_idx])]
    ranked = pvalues[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out[order] = np.clip(adj, 0, 1)
    return out


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

def test_optimization(trajectories: np.ndarray, agg: str = "mean", stat: str = "median",
                       n_perm: int = 9999, alpha: float = 0.05,
                       rng: np.random.Generator | None = None,
                       categories: dict = None
    ) -> pd.DataFrame:
    """
    trajectories: (n_problems, n_samples, n_runs, n_steps) array, NaN for
    missing samples/steps.

    For each problem:
      1. Aggregates samples per step (agg='mean' or 'median').
      2. For each run, computes a robust slope of (value - value at that
         run's first valid step) vs. steps-since-start.
      3. Runs a sign-flip permutation test on the resulting per-run slopes
         to test whether they are systematically above zero.

    Returns a DataFrame indexed by problem with:
      observed_slope : median (or mean) per-run slope for that problem
      p_value        : sign-flip test p-value
      p_value_BH     : Benjamini-Hochberg corrected p-value across problems
      n_runs_used    : number of runs with enough valid steps to fit a slope
      improves       : True if p_value_BH < alpha AND observed_slope > 0
    """
    if rng is None:
        rng = np.random.default_rng()

    agg_traj = _aggregate_samples(trajectories, agg=agg)  # (n_problems, n_runs, n_steps)
    n_problems = agg_traj.shape[0]

    rows = []
    for p in range(n_problems):
        slopes = _problem_slopes(agg_traj[p])
        res = _sign_flip_test(slopes, n_perm=n_perm, stat=stat, rng=rng)
        res["problem"] = p
        rows.append(res)

    out = pd.DataFrame(rows).set_index("problem")
    out["p_value_BH"] = _benjamini_hochberg(out["p_value"].values)
    out["improves"] = (out["p_value_BH"] < alpha) & (out["observed_slope"] > 0)

    if categories:
        idx_to_category = {}
        for name, indices in categories.items():
            for idx in indices:
                idx_to_category[idx] = name
        out["category"] = out.index.map(idx_to_category)
        return out[["category", "observed_slope", "p_value", "p_value_BH", "n_runs_used", "improves"]]
    return out[["observed_slope", "p_value", "p_value_BH", "n_runs_used", "improves"]]