"""Simple plots for `fit_and_forecast` results."""

import hashlib
import os
import re
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# Blue -> purple -> pink -> red, staying clear of coolwarm's washed-out white midpoint
# so adjacent cutoffs stay visually distinct.
_CUTOFF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cutoff_colors", ["#1f4e9c", "#1fa0a0", "#6a3d9a", "#c81fa0", "#e6293a", "#f2a71b"]
)

_BIG_FONT = 20
_BIG_TITLE_FONT = 24
_BIG_LEGEND_FONT = 18

_VIRIDIS_LINE_MAX = 0.85


def _save_fig(fig, title, default_name, save=True):
    """Save `fig` to `plots/<slug>.pdf` when `save` -- `<slug>` is `title` sanitized to a safe
    filename if given, else `default_name`. Creates the `plots/` directory if it doesn't exist.
    """
    if not save:
        return
    os.makedirs("plots", exist_ok=True)
    slug = re.sub(r"[^\w\-]+", "_", title).strip("_") if title else default_name
    fig.savefig(f"plots/{slug}.pdf", bbox_inches="tight")


def plot_fitted_curves(
    df,
    data,
    family,
    categories,
    cutoff="max",
    y_label="value",
    title=None,
    y_range=None,
    log_x=False,
    save=True,
):
    """One subplot per category: each behaviour's observed points overlaid with its fitted `family` curve.

    y_range : optional (ymin, ymax) tuple to restrict the displayed y-axis range; defaults to
    `ax.set_ylim(bottom=0)` (autoscaled top) when not given.
    log_x : if True, plot the x-axis (step) on a log scale.
    """
    sub_all = df[
        (df["family"] == family)
        & (df["cutoff"] == cutoff)
        & (df["row_id"] != "aggregate")
        & df["converged"]
    ]
    t = np.arange(data.shape[1])

    ncols = 2
    nrows = -(-len(categories) // ncols)

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(9 * ncols, 6.5 * nrows), sharex=True, sharey=True
        )
        axes = np.atleast_2d(axes)

        for ax, (cat_name, idxs) in zip(axes.ravel(), categories.items(), strict=False):
            sub = sub_all[sub_all["row_id"].astype(int).isin(idxs)]
            n = len(sub)
            colors = [plt.cm.viridis(i / max(n - 1, 1) * _VIRIDIS_LINE_MAX) for i in range(n)]
            for color, (_, row) in zip(colors, sub.iterrows(), strict=False):
                ax.scatter(t, data[int(row["row_id"])], color=color, alpha=0.5, s=10, zorder=2)
                ax.plot(t, row["predicted_curve"], color=color, linewidth=1.4, zorder=3)

            # stats = (
            #     f"mean in-sample RMSE = {sub['in_sample_rmse'].mean():.4f}\n"
            #     f"var in-sample RMSE = {sub['in_sample_rmse'].var():.6f}"
            # )
            # ax.text(
            #     0.98, 0.02, stats, transform=ax.transAxes, va="bottom", ha="right",
            #     fontsize=8, bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
            # )
            ax.set_title(_prettify(cat_name), fontsize=_BIG_TITLE_FONT)
            if log_x:
                ax.set_xscale("log")
            if y_range is not None:
                ax.set_ylim(*y_range)
            else:
                ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
            ax.tick_params(direction="out", length=5)

        for ax in axes.ravel()[len(categories) :]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(_prettify(y_label), fontsize=_BIG_FONT)
        for ax in axes[-1, :]:
            ax.set_xlabel("Step", fontsize=_BIG_FONT)

        fig.suptitle(title, fontsize=_BIG_TITLE_FONT)
        fig.tight_layout(pad=1.4, w_pad=0.6)
    _save_fig(fig, title, "fitted_curves", save)
    plt.close(fig)
    return fig


def plot_overlaid_by_category(
    scores, categories, y_label="value", title=None, y_range=None, log_x=False, save=True
):
    """One subplot per category: every behaviour's raw trajectory overlaid, plus its category mean.

    scores : (n_behaviours, n_steps) array, one row per behaviour -- e.g. `success_prob`
        or `avg_harmfulness` from `load_forecast_run`.
    categories : dict[str, list[int]] mapping a category name to the behaviour indices
        (into axis 0 of `scores`) belonging to it, same shape as `plot_overlaid_runs`'s
        `categories` argument.

    `plot_overlaid_runs` puts every behaviour from every category on one shared axis,
    which is illegible once there are a few dozen lines. This splits that single overlay
    into one panel per category (colored via `_viridis_category_colors`, so colors match
    `plot_overlaid_runs`), so each panel only holds that category's own lines and
    individual behaviour trajectories stay readable.
    """
    from thesis.analysis.analyse import _viridis_category_colors

    scores = np.asarray(scores)
    n_beh = scores.shape[0]
    _, legend = _viridis_category_colors(n_beh, categories)
    cat_colors = dict(legend)

    t = np.arange(scores.shape[1])
    ncols = 2
    nrows = -(-len(categories) // ncols)

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(9 * ncols, 6.5 * nrows), sharex=True, sharey=True
        )
        axes = np.atleast_2d(axes)

        for ax, (cat_name, idxs) in zip(axes.ravel(), categories.items(), strict=False):
            color = cat_colors[cat_name]
            sub = scores[idxs]
            has_data = ~np.all(np.isnan(sub), axis=0)
            for row in sub:
                ax.plot(t[has_data], row[has_data], color=color, alpha=0.4, linewidth=1.4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(sub[:, has_data], axis=0)
            ax.plot(t[has_data], mean, color="black", linewidth=2.6)
            ax.set_title(_prettify(cat_name), fontsize=_BIG_TITLE_FONT)
            if y_range is not None:
                ax.set_ylim(*y_range)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
            ax.tick_params(direction="out", length=5)

        if log_x:
            ax.set_xscale("log")
        for ax in axes.ravel()[len(categories) :]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(y_label, fontsize=_BIG_FONT)
        for ax in axes[-1, :]:
            ax.set_xlabel("Step", fontsize=_BIG_FONT)

        fig.suptitle(title, fontsize=_BIG_TITLE_FONT)
        fig.tight_layout(pad=1.4, w_pad=0.6)
    _save_fig(fig, title, "overlaid_by_category", save)
    plt.close(fig)
    return fig


def plot_aggregate_fit(
    df, data, families=None, cutoff="max", y_label="value", title=None, metrics=("in_sample_rmse",)
):
    """Single panel: aggregate's observed points overlaid with each fitted `families` curve."""
    sub_all = df[(df["row_id"] == "aggregate") & (df["cutoff"] == cutoff) & df["converged"]]
    if families is not None:
        sub_all = sub_all[sub_all["family"].isin(families)]
    else:
        families = sub_all["family"].unique()

    agg = np.nanmean(data, axis=0)
    t = np.arange(data.shape[1])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(t, agg, color="black", alpha=0.5, s=10, label="aggregate")

    for i, (_, row) in enumerate(sub_all.iterrows()):
        color = family_color(row["family"])  # fixed per family, shared across all plots
        ax.plot(
            t, row["predicted_curve"], color=color, linewidth=1.5, label=_prettify(row["family"])
        )
        values = ((m, row.get(m)) for m in metrics)
        label = "\n".join(
            f"{_prettify(m)}={v:.4f}" for m, v in values if v is not None and not np.isnan(v)
        )
        if not label:
            continue
        ax.annotate(
            label,
            xy=(t[-1], row["predicted_curve"][-1]),
            xytext=(15, 20 * i - 10 * (len(sub_all) - 1)),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5},
        )

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Step")
    ax.legend(loc="lower right")

    fig.suptitle(title or "Fitted families on aggregate")
    fig.subplots_adjust(right=0.78)
    # plt.close(fig)
    return fig


def _any_sample_success(success):
    return np.nanmax(success, axis=2)  # (problems, runs, steps): did any sample succeed this step?


def _shuffle_steps_per_run(any_sample, rng):
    n_problems, n_runs = any_sample.shape[0], any_sample.shape[1]
    shuffled = any_sample.copy()
    for p in range(n_problems):
        for r in range(n_runs):
            valid = np.flatnonzero(~np.isnan(any_sample[p, r]))
            shuffled[p, r, valid] = any_sample[p, r, rng.permutation(valid)]
    return shuffled


def _cummax_trajectory(any_sample):
    ever_valid = np.maximum.accumulate(~np.isnan(any_sample), axis=-1)  # bool, NaN-free
    cum_success = np.maximum.accumulate(
        np.where(np.isnan(any_sample), 0.0, any_sample), axis=-1
    )  # NaN-free
    valid_count = ever_valid.sum(axis=1)  # (problems, steps)
    total = np.where(ever_valid, cum_success, 0.0).sum(axis=1)  # (problems, steps)
    return np.where(valid_count > 0, total / np.maximum(valid_count, 1), np.nan)


def _per_problem_trajectories(any_sample, shuffled_any_sample):
    observed = _cummax_trajectory(any_sample)

    # P(any sample succeeds) at this step, averaged over runs -- np.nanmean avoided here too,
    # same reasoning as `_cummax_trajectory`.
    valid = ~np.isnan(any_sample)
    valid_count = valid.sum(axis=1)  # (problems, steps)
    per_step = np.where(
        valid_count > 0,
        np.where(valid, any_sample, 0.0).sum(axis=1) / np.maximum(valid_count, 1),
        np.nan,
    )
    per_step_ever_valid = np.maximum.accumulate(~np.isnan(per_step), axis=1)
    independent = np.where(per_step_ever_valid, 1 - np.nancumprod(1 - per_step, axis=1), np.nan)

    shuffled = _cummax_trajectory(shuffled_any_sample)

    return observed, independent, shuffled


def _cumulative_asr_trajectories(any_sample, shuffled_any_sample):
    """`_per_problem_trajectories`, aggregated (mean) over problems to shape (steps,).

    Called once per bootstrap draw, so -- same reasoning as `_cummax_trajectory` -- this avoids
    `np.nanmean`'s per-call copy overhead in favor of a plain sum/count.
    """
    aggregated = []
    for traj in _per_problem_trajectories(any_sample, shuffled_any_sample):
        valid = ~np.isnan(traj)
        count = valid.sum(axis=0)
        aggregated.append(
            np.where(
                count > 0, np.where(valid, traj, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan
            )
        )
    return tuple(aggregated)


def _fit_power_law(curve, ks, eps=1e-6):
    """Fit `-log(curve) = a * ks**-b` by least squares in log-log space (an exact linear solve).

    As `curve` (e.g. cumulative ASR) approaches 1, `-log(curve)` approaches 0 like `(1 - curve)`
    -- so this is testing whether the remaining "distance to 1" shrinks as a power law in `ks`,
    the standard check for this shape of curve. Fitting in log-log space rather than nonlinear
    least-squares on `curve` directly avoids the large-magnitude early points dominating the fit.

    Points with `curve <= eps` or `>= 1 - eps` are dropped, not clipped -- clipping would
    fabricate values that then dominate the regression.

    Returns None if fewer than 3 points survive; otherwise a dict with `a`, `b`, `r2`, `mask`
    (into `curve`/`ks`), `fitted_curve` and `fitted_neg_log` (both length `len(ks)`), `resid`
    (in log-log space, aligned with `mask`), and `b_split_half` (b fit separately on the first vs.
    second half of the usable points, or None if fewer than 6 -- a quick stability check).
    """
    mask = (curve > eps) & (curve < 1 - eps)
    if mask.sum() < 3:
        return None
    log_ks = np.log(ks)
    neg_log = -np.log(curve)

    slope, intercept = np.polyfit(log_ks[mask], np.log(neg_log[mask]), 1)
    b, a = -slope, np.exp(intercept)
    fitted_neg_log = a * ks ** (-b)
    fitted_curve = np.exp(-fitted_neg_log)

    pred = np.log(a) - b * log_ks[mask]
    resid = np.log(neg_log[mask]) - pred
    r2 = 1 - resid.var() / np.log(neg_log[mask]).var()

    b_split_half = None
    h = mask.sum() // 2
    if h >= 3:
        lk, lnla = log_ks[mask], np.log(neg_log[mask])
        b_first = -np.polyfit(lk[:h], lnla[:h], 1)[0]
        b_second = -np.polyfit(lk[h:], lnla[h:], 1)[0]
        b_split_half = (b_first, b_second)

    return {
        "a": a,
        "b": b,
        "r2": r2,
        "mask": mask,
        "fitted_curve": fitted_curve,
        "fitted_neg_log": fitted_neg_log,
        "resid": resid,
        "b_split_half": b_split_half,
    }


def plot_asr_forecast_assumptions(success, n_bootstrap=1000, ci=0.95, eps=1e-6, seed=0, title=None):
    """Cumulative ASR under progressively stronger independence assumptions, with a power-law check.

    A 3x3 grid. Rows, top to bottom: the observed cumulative ASR, what it would be if steps
    were independent (but using the real, rising per-step success rate), and the observed
    successes again but with each run's step order permuted -- a permutation-test control, see
    `_per_problem_trajectories` for the exact calculation of each row's curve.

    Columns, left to right, for each row's curve:
    1. The curve itself: individual behaviours (grey), the aggregate with its bootstrap CI band,
       and the fitted power-law curve from column 2 overlaid for reference.
    2. The power-law check: `-log(curve)` vs. step in log-log space (see `_fit_power_law`) --
       a straight line here means the curve's approach to 1 follows a power law; systematic
       curvature means it doesn't.
    3. Fit residuals (log-log space) vs. log(step), to spot curvature the R^2 alone might hide.

    success : (problems, runs, samples, steps) array of 0/1/NaN per-sample successes,
    e.g. `success = np.where(np.isnan(all_scores), np.nan, all_scores > threshold)`.

    Each curve is the point estimate on the full data; shaded bands are `ci`-percentile
    bootstrap intervals from resampling runs (with replacement) `n_bootstrap` times -- the same
    draws are reused to fit a power law to each replicate, giving the fit-parameter and
    fit-curve bands in column 2 without any extra resampling. Runs are reduced to a per-run
    summary once up front (see `_any_sample_success`), since each draw only needs to re-index
    that, not rescan the full array. The `shuffled` row's per-run step permutation is likewise
    fixed once up front (see `_shuffle_steps_per_run`): resampling still only decides *which*
    runs a draw includes, never re-shuffles a run's own step order.

    The column-1 bands tend to sit mostly *below* the `independent` line rather than straddling
    it, especially at later steps -- this is expected, not a bug. `independent` compounds a
    per-step probability across steps via a running product, which is concave in that
    probability; by Jensen's inequality, the point estimate (fit once on the pooled data) sits
    above the typical bootstrap replicate, and the gap grows the more steps get compounded.
    `observed` and `shuffled` have no such compounding assumption, so their bands stay symmetric.
    """
    n_runs, n_steps = success.shape[1], success.shape[3]
    any_sample = _any_sample_success(success)
    rng = np.random.default_rng(seed)
    shuffled_any_sample = _shuffle_steps_per_run(any_sample, rng)

    per_problem = _per_problem_trajectories(
        any_sample, shuffled_any_sample
    )  # each (problems, steps)
    point = tuple(np.nanmean(traj, axis=0) for traj in per_problem)

    draws = np.empty((3, n_bootstrap, n_steps))
    for b in range(n_bootstrap):
        idx = rng.integers(n_runs, size=n_runs)
        draws[:, b] = _cumulative_asr_trajectories(any_sample[:, idx], shuffled_any_sample[:, idx])
    lo, hi = np.nanpercentile(
        draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=1
    )  # each (3, steps)

    row_names = ["Observed", "If steps were independent", "Same successes, steps shuffled"]
    row_colors = [_CUTOFF_CMAP(x) for x in (0.0, 0.55, 1.0)]
    fit_color = "#e08214"  # one consistent "this is a fit, not data" color across all rows
    t = np.arange(n_steps)
    ks = t + 1.0  # log-log fit needs steps >= 1; offset only used for the fit and columns 2-3

    with plt.rc_context(
        {"font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "axes.linewidth": 0.8}
    ):
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharey="col")
        for r in range(1, 3):  # column 1 (Step): shared within itself
            axes[r, 0].sharex(axes[0, 0])
        for ax in (
            axes[0, 2],
            axes[1, 1],
            axes[1, 2],
            axes[2, 1],
            axes[2, 2],
        ):  # columns 2-3 (step k): one shared axis
            ax.sharex(axes[0, 1])
        for row, (name, color, behaviours, curve, lo_i, hi_i, draws_i) in enumerate(
            zip(row_names, row_colors, per_problem, point, lo, hi, draws, strict=True)
        ):
            fit = _fit_power_law(curve, ks, eps)

            # Reuse the runs-bootstrap draws already computed above: their own -log(curve) gives
            # an empirical CI band for column 2, and refitting each one gives a fit-parameter (and
            # so fit-curve) CI band, at no extra resampling cost.
            neg_log_draws = np.where(
                (draws_i > eps) & (draws_i < 1 - eps), -np.log(draws_i), np.nan
            )
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", RuntimeWarning
                )  # all-NaN steps (e.g. before eps threshold ever holds)
                nla_lo, nla_hi = np.nanpercentile(
                    neg_log_draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0
                )
            boot_fits = [_fit_power_law(draws_i[b], ks, eps) for b in range(n_bootstrap)]
            boot_ab = np.array([(f["a"], f["b"]) for f in boot_fits if f is not None])

            f_lo = f_hi = b_lo = b_hi = None
            if len(boot_ab) >= 2:
                fit_band = boot_ab[:, [0]] * ks[None, :] ** (-boot_ab[:, [1]])
                f_lo, f_hi = np.percentile(
                    fit_band, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0
                )
                b_lo, b_hi = np.percentile(boot_ab[:, 1], [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])

            if fit is None:
                print(f"[{name}] power-law fit skipped: fewer than 3 usable points (eps={eps:g})")
            else:
                b_ci = f"[{b_lo:.4f}, {b_hi:.4f}]" if b_lo is not None else "[n/a]"
                sh = fit["b_split_half"]
                sh_str = f"  split-half b: first={sh[0]:.4f} second={sh[1]:.4f}" if sh else ""
                print(
                    f"[{name}] a={fit['a']:.4g}  b={fit['b']:.4f} {b_ci}  R²={fit['r2']:.4f}  "
                    f"n_fit={fit['mask'].sum()}  n_boot_fits={len(boot_ab)}{sh_str}"
                )

            ax1, ax2, ax3 = axes[row]

            # The fit line/band are analytical curves defined for every k regardless of data
            # availability -- clip them (and, in column 2, the axis) to start where real data
            # starts, so no panel implies the fit is informed by steps that were never recorded.
            # Extrapolating a power-law fit backwards past where it was actually fit can produce
            # extreme values (here, close enough to 0 to look like the curve is forced through
            # zero), which is misleading rather than informative.
            first_valid = int(np.argmax(fit["mask"])) if fit is not None else None

            # Column 1: the curve itself
            # Individual behaviours as faint, uniform-gray background texture -- with 40 of them,
            # coloring each distinctly (or by the row's accent color) would compete with the
            # aggregate line/band for attention rather than just showing the spread of shapes.
            for behaviour_row in behaviours:
                ax1.plot(t, behaviour_row, color="#999999", alpha=0.25, linewidth=0.6, zorder=1)
            ax1.fill_between(t, lo_i, hi_i, color=color, alpha=0.25, linewidth=0, zorder=2)
            ax1.plot(t, curve, color=color, linewidth=2, zorder=3, label="empirical")
            if fit is not None:
                ax1.plot(
                    t[first_valid:],
                    fit["fitted_curve"][first_valid:],
                    color=fit_color,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=4,
                    label=f"power-law fit (b={fit['b']:.2f})",
                )
            ax1.set_ylim(0, 1)
            ax1.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.5)

            # Column 2: power-law check, log-log space
            ax2.fill_between(
                ks, nla_lo, nla_hi, color=color, alpha=0.2, linewidth=0, zorder=1, label="data CI"
            )
            if fit is not None:
                mask = fit["mask"]
                ks_fit = ks[first_valid:]
                if f_lo is not None:
                    ax2.fill_between(
                        ks_fit,
                        f_lo[first_valid:],
                        f_hi[first_valid:],
                        color=fit_color,
                        alpha=0.2,
                        linewidth=0,
                        zorder=2,
                        label="fit CI",
                    )
                # A line, not markers: this data is recorded every ~10 steps and held flat in
                # between (see `_per_problem_trajectories`), so isolated markers on a log-x axis
                # make the dense vs. sparse regions look like gaps in the data -- there aren't any.
                ax2.plot(
                    ks[mask],
                    -np.log(curve[mask]),
                    "-",
                    color=color,
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=3,
                )
                b_label = f"b={fit['b']:.2f} {b_ci}" if b_lo is not None else f"b={fit['b']:.2f}"
                ax2.plot(
                    ks_fit,
                    fit["fitted_neg_log"][first_valid:],
                    color=fit_color,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=4,
                    label=b_label,
                )
                ax2.set_xlim(left=ks_fit[0])
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.5)

            # Column 3: fit residuals, same step-k x-axis as column 2 (shared above)
            ax3.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=1)
            if fit is not None:
                ax3.plot(
                    ks[fit["mask"]],
                    fit["resid"],
                    "-",
                    color=color,
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=2,
                )
            ax3.set_xscale("log")

            for ax in (ax1, ax2, ax3):
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
                ax.tick_params(direction="out", length=3)

            ax1.set_ylabel(f"{name}\ncumulative ASR", fontsize=9)
            ax2.set_ylabel("-log(ASR)  (log)", fontsize=9)
            ax3.set_ylabel("residual, log space", fontsize=9)
            r2_str = f"{fit['r2']:.3f}" if fit is not None else "n/a"
            # loc="right", every row: matplotlib's left/center/right titles share one baseline, so
            # this sits at the exact same height/size as row 0's loc="left" column header below --
            # and, since it's set unconditionally, at the exact same height as rows 1-2's R^2 too.
            ax2.set_title(rf"$R^2$={r2_str}", fontsize=9, loc="right")
            if row == 0:
                ax1.set_title("Cumulative ASR", fontsize=11, loc="left")
                ax2.set_title("Power-law check", fontsize=11, loc="left")
                ax3.set_title("Fit residuals", fontsize=11, loc="left")

        axes[-1, 0].set_xlabel("Step", fontsize=10)
        for ax in axes[-1, 1:]:
            ax.set_xlabel("Step (log scale)", fontsize=10)

        fig.suptitle(
            title
            or f"Cumulative ASR: observed vs. independence assumptions ({ci:.0%} bootstrap CI)",
            fontsize=12,
        )
        fig.tight_layout()
    return fig


def plot_cutoff_fits(
    df, data, cutoffs, family, y_label="value", title=None, metrics=("in_sample_rmse",)
):
    """Single panel: aggregate's observed points overlaid with `family`'s fitted curve at each `cutoff`.

    Each cutoff gets its own color; a thin dashed vertical line marks where that cutoff falls,
    drawn in the matching color. Curves are identified through a frameless legend rather than
    inline labels, since inline end-of-curve labels collide once several cutoffs' curves
    converge to the same corner (a common case here).
    """
    sub_all = df[(df["row_id"] == "aggregate") & (df["family"] == family) & df["converged"]]
    sub_all = sub_all[sub_all["cutoff"].isin(cutoffs)]

    agg = np.nanmean(data, axis=0)
    t = np.arange(data.shape[1])

    axes_width_in, height_in = 6.5, 5.5

    with plt.rc_context(
        {"font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "axes.linewidth": 0.8}
    ):
        fig, ax = plt.subplots(
            figsize=(axes_width_in + 2.5, height_in)
        )  # generous initial legend allowance
        ax.scatter(
            t,
            agg,
            facecolor="none",
            edgecolor="#555555",
            linewidth=0.7,
            alpha=0.6,
            s=16,
            label="data",
            zorder=2,
        )

        n = len(cutoffs)
        colors = [_CUTOFF_CMAP(i / max(n - 1, 1)) for i in range(n)]
        rows_by_cutoff = {row["cutoff"]: row for _, row in sub_all.iterrows()}
        for color, cutoff in zip(colors, cutoffs, strict=False):
            row = rows_by_cutoff.get(cutoff)
            if row is None:
                continue
            values = ((m, row.get(m)) for m in metrics)
            metric_str = ", ".join(
                f"{m}={v:.3f}" for m, v in values if v is not None and not np.isnan(v)
            )
            label = f"cutoff={cutoff}" + (f" ({metric_str})" if metric_str else "")

            ax.plot(t, row["predicted_curve"], color=color, linewidth=2, label=label, zorder=3)
            if cutoff != "max":  # no fixed position to mark when the fit used the whole trajectory
                ax.axvline(
                    cutoff, color=color, linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.6, zorder=1
                )

        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
        ax.tick_params(direction="out", length=3)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlabel("Step", fontsize=11)
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            fontsize=8.5,
            handlelength=1.5,
        )
        ax.set_title(
            title or f"Fitted {family} curve at different cutoffs", fontsize=12, loc="left", pad=10
        )

        # Keep the plot area a fixed, readable size and grow the *figure* to fit the legend
        # (rather than shrinking the axes for long metric strings, or relying on
        # bbox_inches="tight" at save/display time, which a plain fig.savefig() won't apply).
        fig.canvas.draw()
        legend_width_in = legend.get_window_extent().width / fig.dpi
        left_margin_in, gap_in, right_pad_in = 0.85, 0.25, 0.15
        total_width_in = left_margin_in + axes_width_in + gap_in + legend_width_in + right_pad_in
        fig.set_size_inches(total_width_in, height_in)
        ax.set_position(
            (
                left_margin_in / total_width_in,
                0.11,
                axes_width_in / total_width_in,
                0.78,
            )
        )
    return fig


def plot_behaviours_by_failure_reason(
    df,
    data,
    family,
    cutoff,
    failure_reason=None,
    row_ids=None,
    y_label="value",
    title=None,
    y_range=None,
    save=True,
):
    sub = df[(df["family"] == family) & (df["cutoff"] == cutoff) & (df["row_id"] != "aggregate")]
    if failure_reason is not None:
        sub = sub[sub["failure_reason"] == failure_reason]
    if row_ids is not None:
        wanted = {str(r) for r in row_ids}
        sub = sub[sub["row_id"].isin(wanted)]
    sub = sub.sort_values("row_id", key=lambda s: s.astype(int))

    if sub.empty:
        print(
            f"No rows match family={family!r} cutoff={cutoff!r} "
            f"failure_reason={failure_reason!r} row_ids={row_ids!r}."
        )
        return None

    t = np.arange(data.shape[1])
    n = len(sub)
    colors = [plt.cm.viridis(i / max(n - 1, 1) * _VIRIDIS_LINE_MAX) for i in range(n)]

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, ax = plt.subplots(figsize=(11, 7))

        ys = []
        for color, (_, row) in zip(colors, sub.iterrows(), strict=False):
            y = data[int(row["row_id"])]
            ys.append(y)
            has_data = ~np.isnan(y)
            detail = row["failure_detail"] or row["validation_warning"]
            label = f"row {row['row_id']}" + (f" ({detail})" if detail else "")
            ax.plot(t[has_data], y[has_data], color=color, alpha=0.7, linewidth=1.6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stacked = np.vstack(ys)
            has_any = ~np.all(np.isnan(stacked), axis=0)
            mean = np.nanmean(stacked[:, has_any], axis=0)
        ax.plot(t[has_any], mean, color="black", linewidth=3.0, zorder=5)

        if cutoff != "max":
            ax.axvline(
                cutoff,
                color="#555555",
                linestyle=(0, (4, 3)),
                linewidth=1.5,
                alpha=0.7,
                zorder=1,
                label=f"cutoff={cutoff}",
            )

        if y_range is not None:
            ax.set_ylim(*y_range)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Step", fontsize=_BIG_FONT)
        ax.set_ylabel(y_label, fontsize=_BIG_FONT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
        ax.tick_params(direction="out", length=5)
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            fontsize=_BIG_LEGEND_FONT,
            handlelength=1.5,
        )

        ax.set_title(title, fontsize=_BIG_TITLE_FONT, loc="left", pad=10)

        fig.canvas.draw()
        legend_width_in = legend.get_window_extent().width / fig.dpi
        axes_width_in, height_in = 8.5, 6.5
        left_margin_in, gap_in, right_pad_in = 1.0, 0.3, 0.2
        total_width_in = left_margin_in + axes_width_in + gap_in + legend_width_in + right_pad_in
        fig.set_size_inches(total_width_in, height_in)
        ax.set_position(
            (
                left_margin_in / total_width_in,
                0.12,
                axes_width_in / total_width_in,
                0.76,
            )
        )
    _save_fig(fig, title, "behaviours_by_failure_reason", save)
    return fig


def _converged_rows_for_fit(fit_result, family, cutoff):
    """Non-aggregate, converged rows of `fit_result` for one (family, cutoff) fit.

    Returns the filtered DataFrame plus the behaviours' own row_id as an int array, since
    non-converged rows are dropped and the remaining order need not be positionally aligned
    with `all_scores` -- callers should always index `all_scores` by this array, not by
    position in `sub`.
    """
    sub = fit_result[
        (fit_result["family"] == family)
        & (fit_result["cutoff"] == cutoff)
        & (fit_result["row_id"] != "aggregate")
        & fit_result["converged"]
    ]
    row_indices = sub["row_id"].astype(int).to_numpy()
    return sub, row_indices


def _observed_success_pool(all_scores, row_indices, step, threshold, pool_runs=True):
    """Success pool per behaviour (given by `row_indices`) at `step`.

    If `pool_runs` (default), every run's samples are combined into one bag per behaviour:
    returns (n_pool, n_succ), each shape (len(row_indices),) -- observed pool size N_b and
    success count n_succ_b (score > `threshold`), NaN-safe.

    If not `pool_runs`, each run's samples are kept separate (no pooling across runs): returns
    (n_pool, n_succ), each shape (len(row_indices), n_runs) -- one hypergeometric population per
    (behaviour, run), meant to be averaged over afterward rather than combined into one draw.
    """
    scores_at_step = all_scores[row_indices, :, :, step]  # (n_behaviours, n_runs, n_samples)
    success = np.where(np.isnan(scores_at_step), np.nan, (scores_at_step > threshold).astype(float))
    if pool_runs:
        pooled = success.reshape(success.shape[0], -1)  # (n_behaviours, n_runs * n_samples)
        n_pool = (~np.isnan(pooled)).sum(axis=1)
        n_succ = np.nansum(pooled, axis=1).astype(int)
    else:
        n_pool = (~np.isnan(success)).sum(axis=2)  # (n_behaviours, n_runs)
        n_succ = np.nansum(success, axis=2).astype(int)
    return n_pool, n_succ


def _hypergeom_success_curve(n_pool, n_succ, k_values):
    """P(>=1 success in k draws without replacement), NaN past each entry's own pool size.

    n_pool, n_succ : same shape, e.g. (n_behaviours,) when pooling runs, or
        (n_behaviours, n_runs) when runs are kept separate (see `_observed_success_pool`).
    Returns shape (*n_pool.shape, len(k_values)).
    """
    from scipy.stats import hypergeom

    flat_pool, flat_succ = n_pool.ravel(), n_succ.ravel()
    curve = np.full((flat_pool.size, len(k_values)), np.nan)
    for i in range(flat_pool.size):
        if flat_pool[i] == 0:
            continue
        valid_k = k_values <= flat_pool[i]
        p_zero = hypergeom.pmf(0, flat_pool[i], flat_succ[i], k_values[valid_k])
        curve[i, valid_k] = 1 - p_zero
    return curve.reshape(*n_pool.shape, len(k_values))


def _observed_success_curve(n_pool, n_succ, k_values, hypergeometric=True):
    """Per-(behaviour[, run]) observed ASR@k', shape (*n_pool.shape, len(k_values)).

    hypergeometric=True (default): the exact tail for k' draws *without replacement* from that
    entry's own finite pool -- see `_hypergeom_success_curve`. NaN past that entry's own pool
    size (can't draw more than you have).

    hypergeometric=False: the same `1 - (1 - p)**k'` formula the Forecasted side uses, with p
    fixed at that entry's *empirical* success rate (n_succ/n_pool) -- independent draws *with*
    replacement, extrapolated to any k' with no pool-size cap. Isolates how much of a
    Forecasted-vs-Observed gap comes from the fitted vs. empirical probability alone, separate
    from the with- vs. without-replacement sampling assumption.
    """
    if hypergeometric:
        return _hypergeom_success_curve(n_pool, n_succ, k_values)
    p_obs = np.where(n_pool > 0, n_succ / np.maximum(n_pool, 1), np.nan)
    return 1 - (1 - p_obs[..., None]) ** k_values


def plot_asr_forecast_vs_true(
    fit_result, all_scores, family, fit_cutoff, step, threshold=0.5, ax=None
):
    """At a single attack step, compare forecasted ASR-vs-k against ASR-vs-k computed empirically
    from pooled observed samples at that step.

    Forecasted: per-behaviour p_hat(step) is read off `fit_result`'s already-evaluated
    `predicted_curve` (fit at `fit_cutoff`, family `family`), then averaged across behaviours of
    1 - (1 - p_hat)**k -- the probability at least one of k independent draws (at that behaviour's
    forecasted success rate) succeeds.

    Observed: for each behaviour, pools every run's samples at `step` into one bag of size N_b
    (n_succ_b of them successes, i.e. score > `threshold`), then uses the exact hypergeometric
    tail P(>=1 success in k draws without replacement from that bag) -- undefined for k > N_b, left
    NaN there. Averaged across behaviours with `np.nanmean`.

    The shared k-axis runs from 1 to min(N_b) across behaviours, so every behaviour contributes to
    every point on the observed curve (this collapses to `all_scores.shape[1] * all_scores.shape[2]`,
    the full pooled size, when no behaviour is missing samples at this step).

    all_scores : (n_behaviours, n_runs, n_samples, num_steps) raw score array (NaN where absent),
    e.g. as loaded by `load_experiment` -- not the flattened/averaged format `fit_result` was fit on.
    """
    sub, row_indices = _converged_rows_for_fit(fit_result, family, fit_cutoff)
    if sub.empty:
        print(f"No converged fits for family={family!r}, cutoff={fit_cutoff!r}; skipping plot.")
        return None

    p_hat = np.clip(np.array([row["predicted_curve"][step] for _, row in sub.iterrows()]), 0.0, 1.0)

    # --- Observed: pool runs & samples per behaviour at this step (same behaviours as p_hat) ---
    n_pool, n_succ = _observed_success_pool(all_scores, row_indices, step, threshold)

    if n_pool.size and n_pool.max() != n_pool.min():
        print(
            f"Warning: observed pool size N_b varies across behaviours at step={step} "
            f"(min={n_pool.min()}, max={n_pool.max()}); the shared k-axis is capped at min(N_b)."
        )

    k_max = int(n_pool.min()) if n_pool.size and n_pool.min() > 0 else 1
    k_values = np.unique(np.round(np.logspace(0, np.log10(max(k_max, 1)), 200)).astype(int))
    k_values = k_values[(k_values >= 1) & (k_values <= k_max)]

    # --- Predicted: mean over behaviours of 1 - (1 - p_hat)**k ---
    pred_curve = np.mean(1 - (1 - p_hat[:, None]) ** k_values[None, :], axis=0)

    # --- True: per-behaviour hypergeometric tail, NaN past that behaviour's own N_b ---
    true_per_behaviour = _hypergeom_success_curve(n_pool, n_succ, k_values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        true_curve = np.nanmean(true_per_behaviour, axis=0)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    ax.plot(k_values, pred_curve, color="#1f4e9c", linewidth=2, label="Forecasted")
    ax.plot(k_values, true_curve, color="#e6293a", linewidth=2, label="Observed (empirical)")

    n_samples_per_step = all_scores.shape[2]
    if n_samples_per_step <= k_values.max(initial=1):
        ax.axvline(
            n_samples_per_step,
            color="#555555",
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            alpha=0.6,
            zorder=1,
        )
        ax.annotate(
            f"k={n_samples_per_step}\n(per-step draws)",
            xy=(n_samples_per_step, 0.98),
            fontsize=7.5,
            color="#555555",
            va="top",
            ha="left",
        )

    ax.set_xscale("log")
    ax.set_xlim(1, max(k_values.max(initial=1), 2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("k (samples drawn)", fontsize=10.5)
    ax.set_ylabel("ASR@k", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
    ax.tick_params(direction="out", length=3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        f"step={step}  family={family}  fit_cutoff={fit_cutoff}", fontsize=11.5, loc="left", pad=8
    )

    if created_fig:
        fig.tight_layout()
    return fig


def _annotate_avoiding(ax, fig, text, xy, candidates, curve_x, curve_y, avoid_x, **kwargs):
    """`ax.annotate(text, xy=xy, ...)`, picking whichever `(dx, dy, ha, va)` in `candidates`
    (offsets in points, tried in order) is the first to render without its bounding box
    overlapping either the vertical line at `x=avoid_x` or `curve_y` (sampled at `curve_x`,
    both in data coords), *and* without leaving `ax`'s own y-limits -- a candidate above/below
    the axes trivially "clears" the curve (there's nothing to hit up there) but can land on a
    neighboring panel's title instead, which this rules out too. Requires `ax`'s scale/limits to
    already be set to their final values -- the overlap check depends on the real
    data<->display transform. Scores each candidate on continuous badness measures (how far
    outside the axes, what fraction of the curve's sampled points fall inside the label, whether
    it crosses the guide line) and keeps the lowest-scoring one, so it can prefer a distant
    candidate that's mostly clear over a close one that's fully stuck on a locally flat curve.

    Only ever redraws the canvas twice (once to measure the label's rendered size, since its
    width/height is ~constant across candidate positions; once at the very end for the winning
    placement) -- every candidate in between is scored by transforming its analytically-computed
    display-space bbox back to data coordinates, not by actually redrawing it. A naive
    redraw-per-candidate approach is the dominant cost of this function by far and isn't needed
    just to get a bbox size.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    curve_x, curve_y = np.asarray(curve_x), np.asarray(curve_y)
    xlo, xhi = sorted(ax.get_xlim())
    ylo, yhi = sorted(ax.get_ylim())

    probe = ax.annotate(
        text, xy=xy, xytext=(0, 0), textcoords="offset points", ha="center", va="center", **kwargs
    )
    fig.canvas.draw()
    width, height = probe.get_window_extent(renderer=renderer).size  # display (pixel) units
    probe.remove()

    anchor_x, anchor_y = ax.transData.transform(xy)
    pts_to_px = fig.dpi / 72.0
    inv = ax.transData.inverted()

    best = None
    for dx, dy, ha, va in candidates:
        cx, cy = anchor_x + dx * pts_to_px, anchor_y + dy * pts_to_px
        x0d = cx if ha == "left" else (cx - width if ha == "right" else cx - width / 2)
        y0d = cy if va == "bottom" else (cy - height if va == "top" else cy - height / 2)
        (x0, y0), (x1, y1) = inv.transform([(x0d, y0d), (x0d + width, y0d + height)])
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        hits_vline = x0 <= avoid_x <= x1
        in_x_range = (curve_x >= x0) & (curve_x <= x1)
        curve_overlap_frac = (
            float(np.mean((curve_y[in_x_range] >= y0) & (curve_y[in_x_range] <= y1)))
            if in_x_range.any()
            else 0.0
        )
        out_of_bounds_amount = (
            max(0.0, xlo - x0) + max(0.0, x1 - xhi) + max(0.0, ylo - y0) + max(0.0, y1 - yhi)
        )
        score = (out_of_bounds_amount, curve_overlap_frac, hits_vline)
        if best is None or score < best[0]:
            best = (score, dx, dy, ha, va)
        if out_of_bounds_amount == 0.0 and curve_overlap_frac < 0.02 and not hits_vline:
            break

    _, dx, dy, ha, va = best
    return ax.annotate(
        text, xy=xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **kwargs
    )


def _compute_optimal_frontier(grid, k_values, s_values, cost_ratio, min_improvement=None):
    """Cheapest-to-most-expensive scan over `grid` (shape (len(k_values), len(s_values))),
    tracking the best ASR seen so far at each cost level.

    cost(s, k) = cost_ratio * s + k -- attack steps cost `cost_ratio` sampling steps each.
    Produces a step-function frontier (flat while the running-best point doesn't change,
    jumping when a cheaper-or-equal-cost point unlocks higher ASR); consecutive repeated
    points are deduplicated before returning.

    min_improvement : if given, the scan stops as soon as a would-be update's relative gain
    over the current best, `(new - old) / old`, falls below this fraction -- i.e. the frontier
    is cut off once it stops meaningfully improving, rather than continuing (possibly flat, on
    smooth/monotonic data like the forecast) all the way to the most expensive grid point.

    Returns (frontier_s, frontier_k, best_asr) -- `best_asr` is the ASR at the frontier's last
    (highest-cost) point, i.e. the best value the scan found before stopping.
    """
    cost = cost_ratio * s_values[None, :] + k_values[:, None]  # (len(k_values), len(s_values))
    flat_cost, flat_asr = cost.ravel(), grid.ravel()
    order = np.argsort(flat_cost)

    running_best_asr = -np.inf
    running_best_point = None
    frontier_s, frontier_k = [], []
    for idx in order:
        a = flat_asr[idx]
        if np.isnan(a):
            continue
        if a > running_best_asr:
            if (
                min_improvement is not None
                and np.isfinite(running_best_asr)
                and running_best_asr > 0
            ):
                improvement = (a - running_best_asr) / running_best_asr
                if improvement < min_improvement:
                    break
            running_best_asr = a
            i, j = np.unravel_index(idx, grid.shape)
            running_best_point = (s_values[j], k_values[i])
        if running_best_point is not None:
            frontier_s.append(running_best_point[0])
            frontier_k.append(running_best_point[1])

    frontier_s, frontier_k = np.array(frontier_s), np.array(frontier_k)
    if len(frontier_s) == 0:
        return frontier_s, frontier_k, np.nan
    keep = np.ones(len(frontier_s), dtype=bool)
    keep[1:] = (frontier_s[1:] != frontier_s[:-1]) | (frontier_k[1:] != frontier_k[:-1])
    frontier_s, frontier_k = frontier_s[keep], frontier_k[keep]
    return frontier_s, frontier_k, running_best_asr


def _frontier_point_at_cost(
    frontier_s, frontier_k, grid, k_values, s_values, cost_ratio, target_cost
):
    """The frontier's own (s, k, ASR) triple at whatever cost is the most expensive one still
    <= `target_cost` -- "what's the best (s, k) you can afford at this compute budget?"

    `frontier_s`/`frontier_k` are `_compute_optimal_frontier`'s stepwise frontier (already in
    increasing-cost order); `grid`/`k_values`/`s_values` supply the ASR value at that exact
    point, looked up by exact match (frontier points are always taken directly from
    `k_values`/`s_values`, so this never needs interpolation).

    Returns None if the frontier is empty, or if even its cheapest point costs more than
    `target_cost` (there's nothing affordable to mark).
    """
    if len(frontier_s) == 0:
        return None
    costs = cost_ratio * frontier_s + frontier_k
    affordable = costs <= target_cost
    if not affordable.any():
        return None
    idx = np.nonzero(affordable)[0][-1]
    s_mark, k_mark = frontier_s[idx], frontier_k[idx]
    k_idx = np.nonzero(k_values == k_mark)[0][0]
    s_idx = np.nonzero(s_values == s_mark)[0][0]
    return s_mark, k_mark, grid[k_idx, s_idx]


_FRONTIER_SMOOTH_WINDOW = 25


def _nanmean_smooth_grid(grid, window):
    """NaN-aware centered moving average along axis=1 (the s/step axis), `window` steps wide --
    used only to compute a less noise-sensitive compute-optimal frontier; the grid the heatmap
    itself renders is never smoothed. `min_periods=1` means a step with at least one non-NaN
    neighbor in its window still gets a value (only all-NaN windows stay NaN).
    """
    import pandas as pd

    return (
        pd.DataFrame(grid.T).rolling(window=window, center=True, min_periods=1).mean().to_numpy().T
    )


def _pivot_probs(probs, row_ids, s_values):
    """Pivot `probs` (long format: one row per behaviour/step with columns row_id, step,
    p_forecast, p_true, n_pool, n_succ) into per-column arrays of shape
    (len(row_ids), len(s_values)), aligned to `row_ids` (row order) and `s_values` (column
    order). A (behaviour, step) combination absent from `probs` becomes NaN in every column, so
    a behaviour missing a value at some step is naturally excluded from that step's average
    rather than needing an explicit convergence flag.
    """
    import pandas as pd

    idx = pd.MultiIndex.from_product([row_ids, s_values], names=["row_id", "step"])
    aligned = probs.set_index(["row_id", "step"])[
        ["p_forecast", "p_true", "n_pool", "n_succ"]
    ].reindex(idx)
    shape = (len(row_ids), len(s_values))
    return {col: aligned[col].to_numpy(dtype=float).reshape(shape) for col in aligned.columns}


def _probs_at_step(probs, step):
    """One row per behaviour at `step`, indexed by row_id -- the `probs`-dataframe equivalent of
    slicing `fit_result`'s `predicted_curve` at a step. Behaviours with no row at `step` are
    simply absent, not filled with a NaN placeholder.
    """
    return probs[probs["step"] == step].set_index("row_id")[
        ["p_forecast", "p_true", "n_pool", "n_succ"]
    ]


def _observed_curve_from_probs(n_pool, n_succ, p_true, k_values, hypergeometric):
    """Per-behaviour observed ASR@k' from already-computed probabilities/counts -- the
    `probs`-dataframe counterpart of `_observed_success_curve` (which instead derived pooled
    counts on the fly from raw `all_scores`).

    hypergeometric=True: exact tail from `n_pool`/`n_succ` (see `_hypergeom_success_curve`) --
    NaN past a behaviour's own pool size, same meaning as elsewhere in this file.
    hypergeometric=False: `1 - (1 - p_true)**k'`, using `p_true` exactly as given -- unlike
    `_observed_success_curve`, this never recomputes p from n_succ/n_pool, so `p_true` can be any
    estimator the caller supplies (not necessarily an empirical pooled rate) -- the whole point
    of the `probs`-based plots, which work for any p_forecast/p_true source.
    """
    if hypergeometric:
        return _hypergeom_success_curve(n_pool, n_succ, k_values)
    return 1 - (1 - p_true[..., None]) ** k_values


def build_asr_probs(fit_result, all_scores, family, fit_cutoff, threshold=0.5, pool_runs=True):
    """Build a `probs` long-format dataframe (row_id, step, p_forecast, p_true, n_pool, n_succ)
    from a `fit_and_forecast` result and raw judge scores, for one (family, fit_cutoff) fit --
    the bridge from the old (fit_result, all_scores, family, fit_cutoff, threshold, pool_runs)
    input model to the `probs`-based plotting functions (`plot_asr_compute_heatmap`,
    `plot_asr_decomposition`, `plot_asr_forecast_vs_observed_scatter_grid`).

    p_forecast comes straight from each converged, non-aggregate row's `predicted_curve`
    (see `_converged_rows_for_fit`), clipped to [0, 1], for every step it was evaluated on.
    p_true/n_pool/n_succ come from pooling `all_scores` (thresholded at `threshold`, pooled
    across runs/samples if `pool_runs`) per (behaviour, step) -- see `_observed_success_pool`.
    A step past `all_scores`'s own step range, or with an empty pool, keeps its p_forecast row
    but gets NaN p_true/n_pool/n_succ (no observed value there, same as the old functions'
    NaN/blank handling).

    Raises ValueError if no rows converged for (family, fit_cutoff).
    """
    import pandas as pd

    sub, row_indices = _converged_rows_for_fit(fit_result, family, fit_cutoff)
    if sub.empty:
        raise ValueError(f"No converged fits for family={family!r}, cutoff={fit_cutoff!r}.")

    n_T = len(sub.iloc[0]["predicted_curve"])
    n_steps_available = all_scores.shape[3]
    n_behaviours = len(row_indices)

    p_forecast = np.clip(
        np.stack([row["predicted_curve"] for _, row in sub.iterrows()]), 0.0, 1.0
    )  # (n_behaviours, n_T)

    p_true = np.full((n_behaviours, n_T), np.nan)
    n_pool_arr = np.full((n_behaviours, n_T), np.nan)
    n_succ_arr = np.full((n_behaviours, n_T), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 0/empty-pool steps, expected
        for step in range(min(n_T, n_steps_available)):
            n_pool, n_succ = _observed_success_pool(
                all_scores, row_indices, step, threshold, pool_runs=pool_runs
            )
            valid = n_pool > 0
            p_true[valid, step] = n_succ[valid] / n_pool[valid]
            n_pool_arr[valid, step] = n_pool[valid]
            n_succ_arr[valid, step] = n_succ[valid]

    row_id_col = np.repeat([str(r) for r in row_indices], n_T)
    step_col = np.tile(np.arange(n_T), n_behaviours)

    return pd.DataFrame(
        {
            "row_id": row_id_col,
            "step": step_col,
            "p_forecast": p_forecast.ravel(),
            "p_true": p_true.ravel(),
            "n_pool": n_pool_arr.ravel(),
            "n_succ": n_succ_arr.ravel(),
        }
    )


def asr_at_k_forecast_error(probs, k, cutoff="max", hypergeometric=True, n_endpoint=5):
    """RMSE and holdout endpoint error between forecasted and observed ASR@k, from a `probs`
    long-format dataframe (row_id, step, p_forecast, p_true, n_pool, n_succ) -- e.g.
    `build_asr_probs`'s or `forecast_success_probs.build_asr_probs_from_fit`'s output.

    ASR@k is built exactly like `plot_asr_compute_heatmap`'s two panels, just at one `k`
    instead of a whole grid: forecasted is `1 - (1-p_forecast)**k` (independent draws with
    replacement); observed is `_observed_curve_from_probs`'s exact without-replacement
    hypergeometric tail from `n_pool`/`n_succ` when `hypergeometric` (default), or the same
    `1-(1-p_true)**k` formula when not.

    cutoff : the fit's cutoff (a step-column index, or "max") -- used only to pick which steps
        count as "held out" for `holdout_endpoint_error`; pass the same cutoff the forecast in
        `probs` was fit with. `"max"` (default) means no holdout, so `holdout_endpoint_error`
        is always None.
    n_endpoint : number of held-out steps (from the end) averaged over for
        `holdout_endpoint_error`, mirroring `forecast.py`'s `holdout_endpoint_error`.

    Returns `(rmse, holdout_endpoint_error)`: both are per-behaviour errors averaged across
    behaviours (so a behaviour with more observed steps doesn't dominate). `holdout_endpoint_
    error` is None if `cutoff == "max"` or no steps beyond it have an observed ASR@k value.
    """
    row_ids = probs["row_id"].unique()
    s_values = np.arange(0, int(probs["step"].max()) + 1)
    pivoted = _pivot_probs(probs, row_ids, s_values)

    p_forecast = np.clip(pivoted["p_forecast"], 0.0, 1.0)  # (n_behaviours, n_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a behaviour missing a step -> NaN there
        forecast_asr = 1 - (1 - p_forecast) ** k
        observed_asr = _observed_curve_from_probs(
            pivoted["n_pool"], pivoted["n_succ"], pivoted["p_true"], np.array([k]), hypergeometric
        )[..., 0]  # (n_behaviours, n_s)

        rmse_per_behaviour = np.sqrt(np.nanmean((forecast_asr - observed_asr) ** 2, axis=1))
        rmse = float(np.nanmean(rmse_per_behaviour))

    holdout_endpoint_error = None
    if cutoff != "max":
        hold_cols = np.flatnonzero((s_values > cutoff) & np.any(~np.isnan(observed_asr), axis=0))
        cols = hold_cols[-n_endpoint:]
        if len(cols) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a behaviour missing all of `cols`
                err = np.abs(forecast_asr[:, cols] - observed_asr[:, cols])
                holdout_endpoint_error = float(np.nanmean(np.nanmean(err, axis=1)))

    return rmse, holdout_endpoint_error


def plot_asr_compute_heatmap(
    probs,
    cost_ratio=5,
    k_max=None,
    k_min=0,
    s_max=None,
    fill_observed_gaps=True,
    smooth_frontier=False,
    match_forecast_steps_to_observed=False,
    observed_hypergeometric=True,
    mark_cost=None,
    title=None,
    save=True,
):
    """Two side-by-side heatmaps of ASR over (attack step s, pooled samples k), each with a
    black compute-optimal frontier overlaid -- Figure 3 style from "Sampling-Aware Adversarial
    Attacks Against Large Language Models". Left panel: forecasted (`1 - (1-p_forecast(s))**k`,
    averaged over behaviours). Right panel: observed (per-behaviour ASR@k from `probs`, averaged
    with `np.nanmean`) -- NaN (blank) wherever a step has no `p_true`/`n_pool` value in `probs`.

    probs : long-format dataframe, one row per (behaviour, step) that has a value -- columns
        row_id, step, p_forecast, p_true, n_pool, n_succ. p_forecast/p_true are per-sample
        success probabilities from whatever source you like (a fitted curve, an empirical rate,
        any other estimator); n_pool/n_succ are the finite sample pool size and success count
        backing p_true at that (behaviour, step), used only for the exact hypergeometric observed
        curve (see observed_hypergeometric below) -- leave them NaN if you only ever want
        observed_hypergeometric=False. A behaviour missing a row at some step is simply excluded
        from that step's average, no convergence flag needed.
    cost_ratio : cost of one attack step relative to one sample (default 5).
    k_max : max samples drawn (y-axis); defaults to the largest `n_pool` seen anywhere in `probs`
        (or 100 if `n_pool` is entirely NaN).
    k_min : min samples drawn (y-axis); defaults to 1. Clipped up to 1 if given <1, and down to
        k_max if given above it.
    s_max : max attack step (x-axis); defaults to the largest `step` present in `probs`.
    fill_observed_gaps : with a sparse sampling_schedule, only a fraction of steps have any
        observed data, so the right panel's dense (s_values-wide) grid is mostly NaN -- each
        real column renders as a near-invisible sliver next to wide blank gaps. When True
        (default), the right panel is instead rendered on just the steps that actually have
        data, letting `pcolormesh` auto-size each cell to fill the gap to its neighbors, so
        the real observations read as solid periodic bands instead of hairlines. This only
        changes what's *drawn*; the compute-optimal frontier is still computed from the true
        per-step `observed_grid` (nothing is invented between real samples).
    smooth_frontier : if True, both panels' compute-optimal frontier is computed from a version
        of its ASR grid smoothed with a NaN-aware, centered moving average over a
        `_FRONTIER_SMOOTH_WINDOW`-step (25) window along the attack-step axis (see
        `_nanmean_smooth_grid`) -- this keeps a handful of noisy empirical ASR points from
        yanking the Observed frontier onto a step that doesn't generalize. Only the frontier
        *path* is computed from the smoothed grid; the rendered heatmap colors are always the
        true, unsmoothed grid. False (default) uses the raw grid, as before. Both panels'
        endpoints are marked with the best ASR reached, printed just above the axes.
    match_forecast_steps_to_observed : the Observed frontier can only ever consider steps that
        actually have samples (e.g. every 10th step, per a sparse sampling_schedule) -- the
        Forecasted frontier, evaluated on the dense fitted curve, has no such restriction and by
        default can pick any step. When True, the Forecasted frontier is instead computed only
        over the same steps the Observed panel actually has data for, so the two are compared
        under the same "how often could you realistically stop and check" budget. Only affects
        which steps the *frontier* may choose; the Forecasted heatmap itself still renders on
        the full dense grid.
    observed_hypergeometric : if True (default), the Observed panel uses the exact hypergeometric
        tail -- draws without replacement from each behaviour's own finite pool (`n_pool`,
        `n_succ`), undefined past that pool's own size. If False, it instead uses the *same*
        `1 - (1-p)**k` formula the Forecasted panel uses, with p fixed at `p_true` -- independent
        draws with replacement, extrapolated to any k with no pool-size cap. Useful for isolating
        how much of a Forecasted-vs-Observed gap is due to the forecast vs. observed probability
        alone, as opposed to the sampling-without-replacement correction.
    mark_cost : if given, a compute budget C (same units as `cost_ratio * s + k`) at which to
        mark each panel's frontier -- the point is the most expensive one the frontier reaches
        without exceeding C (see `_frontier_point_at_cost`), drawn as a gold star on that panel
        with a "C=..." label, and captioned at the bottom of the figure as
        "s=..., k=... (ASR=...)" for both panels side by side -- meant for spotting exactly where
        and how far the Forecasted and Observed frontiers diverge at a given budget, rather than
        just eyeballing the two curves. A panel whose cheapest frontier point already costs more
        than C gets no marker; its caption says so instead. None (default) marks nothing.
    title : optional suptitle override; defaults to "ASR: forecasted vs. observed".

    Returns (fig, (ax_left, ax_right)).
    """
    if probs.empty:
        print("`probs` is empty; skipping plot.")
        return None

    row_ids = probs["row_id"].unique()

    if s_max is None:
        s_max = int(probs["step"].max())
    elif s_max > probs["step"].max():
        print(
            f"s_max={s_max} exceeds the largest step in `probs` ({probs['step'].max()}); nothing past that has data."
        )
    s_values = np.arange(0, s_max + 1)

    if k_max is None:
        pool_max = probs["n_pool"].max()
        k_max = int(pool_max) if np.isfinite(pool_max) else 100
    k_min = max(k_min, 1)
    if k_min > k_max:
        print(f"k_min={k_min} exceeds k_max={k_max}; clipping k_min to k_max.")
        k_min = k_max
    # logspace's first two args are exponents of 10, not the values themselves -- both ends
    # need log10() to actually span [k_min, k_max].
    k_values = np.unique(np.round(np.logspace(np.log10(k_min), np.log10(k_max), 60)).astype(int))
    k_values = k_values[k_values >= k_min]

    pivoted = _pivot_probs(probs, row_ids, s_values)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # behaviours missing a step are all-NaN there

        # --- Forecasted grid: mean over behaviours of 1 - (1 - p_forecast(s))**k, for every (k, s) ---
        p_forecast = np.clip(pivoted["p_forecast"], 0.0, 1.0)  # (n_behaviours, n_s)
        forecast_grid = np.nanmean(
            1 - (1 - p_forecast[None, :, :]) ** k_values[:, None, None], axis=1
        )  # (n_k, n_s)

        # --- Observed grid: per-behaviour ASR@k from `probs`, averaged over behaviours ---
        observed_curve = _observed_curve_from_probs(
            pivoted["n_pool"],
            pivoted["n_succ"],
            pivoted["p_true"],
            k_values,
            observed_hypergeometric,
        )  # (n_behaviours, n_s, n_k)
        observed_grid = np.nanmean(observed_curve, axis=0).T  # (n_k, n_s)

    has_data = ~np.all(np.isnan(observed_grid), axis=0)  # steps the Observed panel has any data for

    # Frontiers are computed from these grids -- the true (sparse) grid by default, or a
    # smoothed copy when `smooth_frontier` (see `_nanmean_smooth_grid`); either way this is only
    # ever the input to the frontier scan below, never what's rendered as the heatmap.
    if smooth_frontier:
        forecast_grid_for_frontier = _nanmean_smooth_grid(forecast_grid, _FRONTIER_SMOOTH_WINDOW)
        observed_grid_for_frontier = _nanmean_smooth_grid(observed_grid, _FRONTIER_SMOOTH_WINDOW)
    else:
        forecast_grid_for_frontier = forecast_grid
        observed_grid_for_frontier = observed_grid

    if match_forecast_steps_to_observed:
        frontier_forecast = _compute_optimal_frontier(
            forecast_grid_for_frontier[:, has_data],
            k_values,
            s_values[has_data],
            cost_ratio,
        )
    else:
        frontier_forecast = _compute_optimal_frontier(
            forecast_grid_for_frontier,
            k_values,
            s_values,
            cost_ratio,
        )
    frontier_observed = _compute_optimal_frontier(
        observed_grid_for_frontier, k_values, s_values, cost_ratio
    )

    vmin = np.nanmin([np.nanmin(forecast_grid), np.nanmin(observed_grid)])
    vmax = np.nanmax([np.nanmax(forecast_grid), np.nanmax(observed_grid)])

    if fill_observed_gaps:
        s_observed_render = s_values[has_data]
        observed_grid_render = observed_grid[:, has_data]
    else:
        s_observed_render = s_values
        observed_grid_render = observed_grid

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(19, 8.5), sharey=True)
        observed_title = "Observed"

        frontier_line = None
        mark_captions = []
        for ax, grid, x_coords, (
            frontier_s,
            frontier_k,
            best_asr,
        ), panel_title, grid_for_frontier in (
            (
                ax_left,
                forecast_grid,
                s_values,
                frontier_forecast,
                "Forecasted",
                forecast_grid_for_frontier,
            ),
            (
                ax_right,
                observed_grid_render,
                s_observed_render,
                frontier_observed,
                observed_title,
                observed_grid_for_frontier,
            ),
        ):
            mesh = ax.pcolormesh(
                x_coords, k_values, grid, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest"
            )
            if grid.shape[0] > 1 and grid.shape[1] > 1:
                ax.contour(
                    x_coords, k_values, grid, levels=15, colors="white", alpha=0.25, linewidths=0.5
                )

            # Set before any annotation below -- the overlap check further down needs the axes'
            # real (final) data<->display transform, not the default autoscaled one.
            ax.set_yscale("log")
            ax.set_xlim(s_values[0], s_values[-1])

            if len(frontier_s):
                (frontier_line,) = ax.plot(frontier_s, frontier_k, color="black", linewidth=2.6)
                # Endpoint marker + best-ASR label, at the top of the plot (not at the line's own
                # height) but x-aligned with where the frontier ends -- x in data coords, y in axes
                # fraction. The triangle sits right at the axes' top edge; the label just above it.
                trans = ax.get_xaxis_transform()
                end_s = frontier_s[-1]
                ax.plot(
                    end_s,
                    1.0,
                    marker="v",
                    color="black",
                    markersize=9,
                    transform=trans,
                    clip_on=False,
                    zorder=5,
                )
                ax.annotate(
                    f"ASR={best_asr:.3f}",
                    xy=(end_s, 1.02),
                    xycoords=trans,
                    fontsize=_BIG_LEGEND_FONT,
                    ha="center",
                    va="bottom",
                    clip_on=False,
                )

            if mark_cost is not None:
                point = _frontier_point_at_cost(
                    frontier_s,
                    frontier_k,
                    grid_for_frontier,
                    k_values,
                    s_values,
                    cost_ratio,
                    mark_cost,
                )
                if point is None:
                    mark_captions.append(
                        f"{panel_title}: C={mark_cost} is below the cheapest frontier point"
                    )
                else:
                    s_mark, k_mark, asr_mark = point
                    ax.scatter(
                        [s_mark],
                        [k_mark],
                        marker="*",
                        s=500,
                        color="#f2a71b",
                        edgecolor="black",
                        linewidth=0.9,
                        zorder=6,
                        clip_on=False,
                    )
                    mark_captions.append(
                        f"{panel_title}: s={s_mark}, k={k_mark}  (ASR={asr_mark:.3f})"
                    )

            ax.set_xlabel("Attack step s", fontsize=_BIG_FONT)
            # pad is generous -- the frontier's "ASR=" / "C=" labels sit just above the data
            # area (axes-fraction y~1.02-1.03) and, at this font size, their own text height can
            # reach almost as high as a tighter pad would put the title, so the title needs
            # enough clearance to stay clear of them rather than just the axes' top spine.
            ax.set_title(panel_title, fontsize=_BIG_TITLE_FONT, pad=34)

        ax_left.set_ylabel("Samples k", fontsize=_BIG_FONT)
        fig.subplots_adjust(
            left=0.08,
            top=0.8,
            right=0.86,
            wspace=0.05,
            bottom=0.16 if mark_cost is not None else 0.1,
        )
        cbar = fig.colorbar(mesh, ax=(ax_left, ax_right), label="ASR", fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=_BIG_LEGEND_FONT)
        cbar.set_label("ASR", fontsize=_BIG_FONT)
        if frontier_line is not None:
            fig.legend(
                [frontier_line],
                ["Compute-optimal frontier"],
                loc="upper center",
                bbox_to_anchor=(0.46, 0.965),
                frameon=False,
                ncols=1,
                fontsize=_BIG_LEGEND_FONT,
            )
        # if mark_cost is not None:
        # Once for the whole figure, in its top-left corner -- the marked cost is the same
        # number in both panels, so a per-panel label was just a duplicate that also kept
        # colliding with the frontier's "ASR=" label or the panel title depending on where
        # the marked point landed. A single figure-level corner label competes with neither.
        # fig.text(
        #     0.01, 0.985, f"C={mark_cost}",
        #     fontsize=_BIG_LEGEND_FONT, fontweight="bold", ha="left", va="top",
        #     bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f2a71b", "edgecolor": "black", "alpha": 0.9},
        # )
        if mark_captions:
            fig.text(
                0.5,
                0.02,
                f"Cost = {cost_ratio}$\\cdot$s + k,  C={mark_cost}:   "
                + "    |    ".join(mark_captions),
                ha="center",
                va="bottom",
                fontsize=_BIG_LEGEND_FONT,
            )
        fig.suptitle(title, fontsize=_BIG_TITLE_FONT, y=0.99)

    _save_fig(fig, title, "asr_compute_heatmap", save)
    return fig, (ax_left, ax_right)


def plot_asr_decomposition(
    probs,
    step,
    k,
    k_max=None,
    n_k_points=200,
    title=None,
    observed_hypergeometric=True,
    show_mean_probability_curve=False,
    save=True,
):
    """Per-behaviour decomposition of the aggregate ASR at one (step, k) point of
    `plot_asr_compute_heatmap` -- same Forecasted/Observed two-panel layout, so a single (s, k)
    cell of that heatmap can be traced back to the individual behaviours it averages over.

    Each panel plots, per behaviour, ASR@k' against k' on a log axis:
    - Forecasted: `1 - (1 - p_forecast(step))**k'`, the same formula the heatmap's left panel grids up.
    - Observed: by default (`observed_hypergeometric=True`) the exact hypergeometric tail from
      that behaviour's own `n_pool`/`n_succ` at `step` (see `_hypergeom_success_curve`), NaN (and
      so visually truncated) past that behaviour's own pool size N_b. With
      `observed_hypergeometric=False`, instead the *same* `1 - (1-p)**k'` formula as Forecasted,
      with p fixed at `p_true` -- no pool-size cap, so lines no longer truncate -- letting you
      isolate how much of the Forecasted-vs-Observed gap comes from the forecast vs. observed
      probability alone, separate from the sampling-without-replacement correction.

    These per-behaviour curves are drawn thin and faint ("the actual forecast", for context) so
    they don't compete with two kinds of markers layered on top, which are the actual answer to
    "where does the aggregate number come from":
    - a small circle at k'=1, i.e. each behaviour's per-sample success probability at `step`
      (`p_forecast` on the left, `p_true` on the right);
    - a diamond at k'=`k`, each behaviour's own contribution to the aggregate ASR@k being decomposed.

    Two more markers at k'=k give two distinct summary statistics of those same diamonds -- the
    number reported as "ASR" is always the mean, never the median, so both are shown as separate
    markers rather than conflated:
    - a black diamond + "ASR=" label at the *mean* of the diamonds -- a bold black line (the mean
      aggregate curve, `nanmean` on the right) passes through this exact point, and it's the
      number each panel's heatmap cell at (step, k) actually shows. The label's position is
      computed at render time by trying candidate offsets until one's actual bounding box clears
      both the black curve and the k'=k guide line, rather than a fixed offset.
    - an orange star (legend only, no text label -- distinguished by color from the red k'=1
      median-per-sample star) at the *median* of the diamonds -- the behaviour that would sit in
      the middle if all behaviours' ASR@k were sorted. A robustness check on the mean, since a
      handful of easy or hard outlier behaviours can pull the mean away from where most
      behaviours actually sit; the two markers landing far apart is itself informative.
    Behaviours are colored by their rank in `p_forecast(step)` (easy -> hard), consistently
    between panels, so the same behaviour can be traced across Forecasted and Observed.

    probs : long-format dataframe, one row per (behaviour, step) that has a value -- columns
        row_id, step, p_forecast, p_true, n_pool, n_succ. See `plot_asr_compute_heatmap` for the
        exact schema; only rows at `step` are used here, so every point lines up 1:1 across
        panels. A behaviour missing a row at `step` is excluded from both panels.
    step : the attack step s to decompose.
    k : the number of samples to decompose ASR@k at; also marks where the diamond markers and
        the vertical guide line sit. Widens `k_max` if needed so `k` itself stays in view.
    k_max : right edge of the k-axis (samples drawn); defaults to the largest `n_pool` among
        behaviours present at `step` (or 100 if none is finite).
    n_k_points : number of log-spaced k' values the per-behaviour and aggregate curves are drawn
        over (not the marker positions, which use `step`/`k` exactly).
    observed_hypergeometric : see above -- toggles the Observed panel between the exact
        without-replacement tail (default) and the with-replacement version of the Forecasted
        panel's own formula, using `p_true`.
    show_mean_probability_curve : if True, adds a dashed line to each panel: the *other* way to
        aggregate that panel's per-behaviour curves -- plug the mean probability into the
        formula once, instead of averaging the curves computed per-behaviour (the solid black
        line, and what the heatmap cell actually shows). Since `1 - (1-p)**k` is concave in p,
        Jensen's inequality guarantees mean(f(p_b)) <= f(mean(p_b)), so this dashed line sits at
        or above the solid one, the gap growing with k and with how much p varies across
        behaviours -- it's never the "more correct" number, just a visual check on how much
        heterogeneity the averaging is hiding. On the Observed side with
        `observed_hypergeometric=True`, the analogous "plug-in" pools every behaviour's
        `n_pool`/`n_succ` into one bag and takes a single hypergeometric curve from that, rather
        than averaging each behaviour's own hypergeometric curve. Off by default.
    title : optional suptitle override; defaults to "ASR decomposition at step=..., k=...".

    Returns (fig, (ax_left, ax_right)).
    """
    sub = _probs_at_step(probs, step)
    if sub.empty:
        print(f"No behaviours with data at step={step}; skipping plot.")
        return None

    k = max(int(k), 1)

    p_hat_s = np.clip(sub["p_forecast"].to_numpy(dtype=float), 0.0, 1.0)
    p_obs_s = sub["p_true"].to_numpy(dtype=float)
    n_pool = sub["n_pool"].to_numpy(dtype=float)
    n_succ = sub["n_succ"].to_numpy(dtype=float)

    if k_max is None:
        pool_max = np.nanmax(n_pool) if n_pool.size else np.nan
        k_max = int(pool_max) if np.isfinite(pool_max) else 100
    k_max = max(k_max, k)
    k_values = np.unique(np.round(np.logspace(0, np.log10(k_max), n_k_points)).astype(int))
    k_values = k_values[k_values >= 1]

    finite_pool = n_pool[np.isfinite(n_pool)]
    if observed_hypergeometric and finite_pool.size and k > finite_pool.max():
        print(
            f"Warning: k={k} exceeds every behaviour's observed pool size at step={step} "
            f"(max N_b={finite_pool.max():.0f}); observed ASR@k is undefined (no diamond markers) on the right panel."
        )
    elif observed_hypergeometric and finite_pool.size and k > finite_pool.min():
        print(
            f"Note: k={k} exceeds some behaviours' observed pool size at step={step} "
            f"(min N_b={finite_pool.min():.0f}); those behaviours have no observed ASR@k diamond and their "
            "line stops early."
        )

    forecast_curves = 1 - (1 - p_hat_s[:, None]) ** k_values[None, :]  # (n_behaviours, n_k)
    forecast_at_k = 1 - (1 - p_hat_s) ** k
    forecast_agg_curve = np.mean(forecast_curves, axis=0)
    forecast_agg_at_k = np.mean(forecast_at_k)
    forecast_median_at_k = np.median(forecast_at_k)

    observed_curves = _observed_curve_from_probs(
        n_pool, n_succ, p_obs_s, k_values, observed_hypergeometric
    )  # (n_behaviours, n_k)
    observed_at_k = _observed_curve_from_probs(
        n_pool, n_succ, p_obs_s, np.array([k]), observed_hypergeometric
    )[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN k' past every behaviour's own N_b
        observed_agg_curve = np.nanmean(observed_curves, axis=0)
        observed_agg_at_k = np.nanmean(observed_at_k)
        observed_median_at_k = np.nanmedian(observed_at_k)

    forecast_plugin_curve = observed_plugin_curve = None
    if show_mean_probability_curve:
        forecast_plugin_curve = 1 - (1 - np.mean(p_hat_s)) ** k_values
        if observed_hypergeometric:
            observed_plugin_curve = _hypergeom_success_curve(
                np.array([np.nansum(n_pool)]), np.array([np.nansum(n_succ)]), k_values
            )[0]
        else:
            observed_plugin_curve = 1 - (1 - np.nanmean(p_obs_s)) ** k_values

    # Rank, not raw value, so colors stay well-spread even when many p_hat(s) cluster together.
    rank = np.argsort(np.argsort(p_hat_s))
    norm_rank = rank / max(len(rank) - 1, 1)
    colors = plt.cm.viridis(norm_rank)

    observed_title = "Observed"
    panels = (
        (
            forecast_curves,
            forecast_at_k,
            forecast_agg_curve,
            forecast_agg_at_k,
            forecast_median_at_k,
            p_hat_s,
            forecast_plugin_curve,
            "Forecasted",
        ),
        (
            observed_curves,
            observed_at_k,
            observed_agg_curve,
            observed_agg_at_k,
            observed_median_at_k,
            p_obs_s,
            observed_plugin_curve,
            observed_title,
        ),
    )

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(19, 8.5), sharex=True, sharey=True)

        for ax, (
            curves,
            at_k,
            agg_curve,
            agg_at_k,
            median_at_k,
            per_sample_p,
            plugin_curve,
            panel_title,
        ) in zip((ax_left, ax_right), panels, strict=True):
            # Set before any placement math below -- the collision check needs the axes' real
            # (final) data<->display transform, not the default linear/unset one.
            ax.set_xscale("log")
            ax.set_xlim(1, k_values.max())
            ax.set_ylim(0, 1)

            for i in range(len(p_hat_s)):
                ax.plot(k_values, curves[i], color=colors[i], alpha=0.3, linewidth=0.9, zorder=1)

            valid_p = ~np.isnan(per_sample_p)
            ax.scatter(
                np.full(valid_p.sum(), 1),
                per_sample_p[valid_p],
                color=colors[valid_p],
                s=110,
                marker="o",
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
                label="per-sample success prob (k=1)",
            )
            valid_k = ~np.isnan(at_k)
            ax.scatter(
                np.full(valid_k.sum(), k),
                at_k[valid_k],
                color=colors[valid_k],
                s=60,
                marker="D",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
                label=f"per-behaviour ASR@{k}",
            )

            ax.axvline(
                k, color="#555555", linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.6, zorder=2
            )
            ax.plot(
                k_values, agg_curve, color="black", linewidth=2.8, zorder=5, label="mean aggregate"
            )
            if plugin_curve is not None:
                ax.plot(
                    k_values,
                    plugin_curve,
                    color="#555555",
                    linestyle="--",
                    linewidth=2.0,
                    zorder=5,
                    label="ASR(mean p)",
                )
            ax.scatter(
                [1],
                [np.nanmedian(per_sample_p)],
                color="red",
                marker="*",
                s=450,
                zorder=6,
                edgecolor="white",
                linewidth=0.8,
                label="median per-sample prob (k=1)",
            )

            # Two distinct metrics at k'=k: the mean (== the black aggregate curve's own value
            # here, what the heatmap cell shows) and the median of the same per-behaviour
            # diamonds. The median gets a marker only (identified via the legend, a different
            # color from the k'=1 median-per-sample star so the two aren't confused); the mean
            # gets the "ASR=" label, placed by actually checking candidate positions against the
            # rendered black curve and the k'=k guide line rather than a fixed offset.
            if not np.isnan(median_at_k):
                ax.scatter(
                    [k],
                    [median_at_k],
                    color="darkorange",
                    marker="*",
                    s=450,
                    zorder=6,
                    edgecolor="white",
                    linewidth=0.8,
                    label=f"median ASR@{k}",
                    clip_on=False,
                )
            if not np.isnan(agg_at_k):
                ax.scatter(
                    [k],
                    [agg_at_k],
                    color="black",
                    marker="D",
                    s=110,
                    zorder=6,
                    edgecolor="white",
                    linewidth=0.8,
                    label=f"ASR@{k}",
                    clip_on=False,
                )
                label_candidates = [
                    (
                        sign * dist * np.cos(np.deg2rad(ang)),
                        dist * np.sin(np.deg2rad(ang)),
                        "left" if sign > 0 else "right",
                        "bottom" if ang > 0 else "top",
                    )
                    for dist in (28, 45, 70, 110, 170, 260)
                    for ang in (35, -35, 60, -60, 15, -15)
                    for sign in (1, -1)
                ]
                _annotate_avoiding(
                    ax,
                    fig,
                    f"ASR={agg_at_k:.3f}",
                    (k, agg_at_k),
                    label_candidates,
                    curve_x=k_values,
                    curve_y=agg_curve,
                    avoid_x=k,
                    fontsize=_BIG_LEGEND_FONT,
                    fontweight="bold",
                    clip_on=False,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                    arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.8, "shrinkB": 6},
                )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#d5d5d5", linewidth=0.6, zorder=0)
            ax.tick_params(direction="out", length=5)
            ax.set_xlabel("k' (samples drawn)", fontsize=_BIG_FONT)
            ax.set_title(panel_title, fontsize=_BIG_TITLE_FONT, loc="left", pad=14)

        ax_left.set_ylabel("ASR@k'", fontsize=_BIG_FONT)
        fig.subplots_adjust(left=0.08, top=0.82, bottom=0.22, right=0.88, wspace=0.05)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=(ax_left, ax_right),
            label=r"behaviour rank by $\hat{p}(s)$  (easy $\rightarrow$ hard)",
            fraction=0.046,
            pad=0.02,
        )
        cbar.ax.tick_params(labelsize=_BIG_LEGEND_FONT)
        cbar.set_label(
            r"behaviour rank by $\hat{p}(s)$  (easy $\rightarrow$ hard)", fontsize=_BIG_FONT
        )

        # One shared legend below both panels, rather than per-axis, so it never sits over the
        # per-behaviour ASR@k diamonds -- colors/labels are identical across panels anyway.
        handles, labels = ax_left.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.46, 0.15),
            frameon=False,
            ncols=3,
            fontsize=_BIG_LEGEND_FONT,
            handlelength=1.5,
        )

        fig.suptitle(
            title,
            fontsize=_BIG_TITLE_FONT,
            y=0.975,
        )

    _save_fig(fig, title, "asr_decomposition", save)
    return fig, (ax_left, ax_right)


_SLEEK_RC = {
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
}

# Computer Modern (LaTeX's own default typeface) is bundled with matplotlib, so this needs no
# external font install. mathtext.fontset="cm" keeps math elements (e.g. R^2, rho) in the same
# family; axes.unicode_minus is off since cmr10 lacks the Unicode minus glyph.
_LATEX_FONT_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
}

# `_SLEEK_RC` with `_LATEX_FONT_RC` layered on top, for figures meant to stand on their own in
# the printed thesis at `plot_overlaid_runs`'s larger text scale.
_SLEEK_LATEX_RC = {**_SLEEK_RC, **_LATEX_FONT_RC, "axes.linewidth": 1.3}

# Text scale to match: base text (ticks/axis labels) at `_BIG_FONT`, titles/suptitles at
# `_BIG_TITLE_FONT`, legend at `_BIG_LEGEND_FONT` (kept a notch below the rest so it doesn't
# dominate the panel, but never below the thesis-wide 18pt floor).
_BIG_FONT = 20
_BIG_TITLE_FONT = 24
_BIG_LEGEND_FONT = 18

# Muted categorical palette (seaborn "deep"), enough distinct hues for ~10 families
# without the harsh primaries of tab10; reads calmly at print size.
_FAMILY_PALETTE = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
    "#8c8c8c",
    "#ccb974",
    "#64b5cd",
]

# Fixed family -> color map, so a curve family reads as the same color in every plot
# here. Hardcoded (rather than derived from `forecast.ALL_FAMILY_NAMES` at import time)
# because that import is best-effort -- if it ever failed in one process but not
# another, names would silently fall back to the hash-based slot below and land on a
# different color than a run where the import succeeded. Order matches
# `forecast.FAMILY_REGISTRY`; any name not listed here (new family, typo, etc.) gets a
# stable palette slot from its hash instead (see `family_color`).
_CANON_FAMILY_NAMES = (
    "exponential",
    "stretched_exponential",
    "power_law_saturating",
    "logistic",
    "power_exponential_crossover",
)

_FAMILY_COLORS = {
    name: _FAMILY_PALETTE[i % len(_FAMILY_PALETTE)] for i, name in enumerate(_CANON_FAMILY_NAMES)
}


def family_color(name):
    """Fixed color for a curve family, identical across every plot in this module.

    Registered families (those in `forecast.ALL_FAMILY_NAMES`) keep the slot assigned
    at import; any other name is given a deterministic palette slot from its hash and
    cached, so it too stays consistent within and across figures.
    """
    name = str(name)
    if name not in _FAMILY_COLORS:
        idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(_FAMILY_PALETTE)
        _FAMILY_COLORS[name] = _FAMILY_PALETTE[idx]
    return _FAMILY_COLORS[name]


# Distinct dash patterns, cycled in lockstep with `_FAMILY_PALETTE`/`_CANON_FAMILY_NAMES` so a
# family keeps the same linestyle everywhere it keeps the same color. Color alone can't be told
# apart where two families' lines land on exactly the same value and fully overlap; pairing each
# color with its own pattern keeps them legible even then, and stays colorblind- and
# print-safe as a bonus.
_FAMILY_LINESTYLES = [
    "-",
    (0, (6, 2)),
    (0, (1, 1)),
    (0, (4, 1.6, 1, 1.6)),
    (0, (5, 1.5, 1, 1.5, 1, 1.5)),
]

_FAMILY_LINESTYLE_MAP = {
    name: _FAMILY_LINESTYLES[i % len(_FAMILY_LINESTYLES)]
    for i, name in enumerate(_CANON_FAMILY_NAMES)
}


def family_linestyle(name):
    """Fixed linestyle for a curve family, identical across every plot in this module --
    the linestyle analog of `family_color`, using the same registered/hash-fallback scheme
    so a name always lands on the same pattern (and the same slot index as its color).
    """
    name = str(name)
    if name not in _FAMILY_LINESTYLE_MAP:
        idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(_FAMILY_LINESTYLES)
        _FAMILY_LINESTYLE_MAP[name] = _FAMILY_LINESTYLES[idx]
    return _FAMILY_LINESTYLE_MAP[name]


_ACRONYMS = ("rmse", "rmsle", "mse", "mae", "mape", "asr", "auc", "roc", "ci", "r2", "ks")


def _prettify(label):
    """'in_sample_rmse' -> 'in sample RMSE', for axis labels, legends and titles."""
    text = str(label).replace("_", " ")
    for acr in _ACRONYMS:
        text = re.sub(rf"\b{acr}\b", acr.upper(), text, flags=re.IGNORECASE)
    return text


def _forecast_vs_observed_points(
    sub, row_indices, all_scores, step, k, threshold, n_T, observed_hypergeometric=True
):
    """Per-behaviour forecasted vs. observed ASR@k at one (step, k), plus rank-based colors --
    the computation shared by the single- and multi-panel scatter plots below, so sweeping
    `step` across panels never numerically drifts from a single-panel call.

    Forecasted: `1 - (1 - p_hat(step))**k` from `sub`'s already-evaluated `predicted_curve`.
    Observed: `_observed_success_curve` on that behaviour's pooled samples at `step` -- the
    exact hypergeometric tail (draws without replacement from that behaviour's own finite pool,
    NaN past its pool size N_b) when `observed_hypergeometric` (default), or the with-replacement
    `1 - (1-p)**k` plug-in using the empirical success rate when not. Same toggle, same meaning,
    as `observed_hypergeometric` in `plot_asr_decomposition`/`plot_asr_compute_heatmap`.

    Returns (step, k, asr_forecast, asr_observed, colors, valid) -- step/k are the
    clamped/int-cast values actually used; `valid` marks behaviours with a defined observed
    value at `step` (no samples pooled, or -- hypergeometric only -- k past that behaviour's own
    pool size). `asr_forecast`/`colors` are defined for all behaviours regardless.
    """
    if step < 0 or step > n_T - 1:
        print(f"step={step} outside the fitted curves' range (0..{n_T - 1}); clipping.")
        step = int(np.clip(step, 0, n_T - 1))
    k = max(int(k), 1)

    p_hat = np.clip(np.array([row["predicted_curve"][step] for _, row in sub.iterrows()]), 0.0, 1.0)
    asr_forecast = 1 - (1 - p_hat) ** k

    n_pool, n_succ = _observed_success_pool(all_scores, row_indices, step, threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # NaN/0-pool entries, expected
        asr_observed = _observed_success_curve(
            n_pool, n_succ, np.array([k]), hypergeometric=observed_hypergeometric
        )[:, 0]

    # Same rank-based coloring as `plot_asr_decomposition`, so points are comparable across plots.
    rank = np.argsort(np.argsort(p_hat))
    norm_rank = rank / max(len(rank) - 1, 1)
    colors = plt.cm.viridis(norm_rank)

    valid = ~np.isnan(asr_observed)
    return step, k, asr_forecast, asr_observed, colors, valid


def _draw_forecast_vs_observed_scatter(ax, asr_forecast, asr_observed, colors, valid, spearmanr):
    """Diagonal + scatter + Spearman annotation for one panel, shared by the single- and
    multi-panel forecast-vs-observed scatter plots. Axis limits/labels/title are left to the
    caller, since those differ between the two (the grid only labels its leftmost panel's y-axis).
    """
    # Faint tint above/below the diagonal -- a visual cue for "forecast under-/over-estimates"
    # that doesn't compete with the points (very low alpha, drawn first).
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#4C72B0", alpha=0.05, zorder=0, linewidth=0)
    ax.fill_between([0, 1], [0, 0], [0, 1], color="#C44E52", alpha=0.05, zorder=0, linewidth=0)

    ax.plot(
        [0, 1],
        [0, 1],
        color="#8c8c8c",
        linestyle=(0, (3, 2)),
        linewidth=1.1,
        zorder=1,
        label="y = x",
    )
    ax.scatter(
        asr_forecast[valid],
        asr_observed[valid],
        color=colors[valid],
        s=48,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
        zorder=3,
    )
    if valid.sum() >= 2:
        rho, _ = spearmanr(asr_forecast[valid], asr_observed[valid])
        ax.text(
            0.03,
            0.97,
            f"Spearman $\\rho$ = {rho:.3f}\nn = {int(valid.sum())}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#1a1a1a",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.grid(True, color="#ebebeb", linewidth=0.6, zorder=0)
    ax.tick_params(direction="out", length=3, labelsize=9.5)


def plot_asr_forecast_vs_observed_scatter(
    fit_result,
    all_scores,
    family,
    fit_cutoff,
    step,
    k,
    threshold=0.5,
    observed_hypergeometric=True,
    title=None,
    ax=None,
):
    """One point per behaviour: forecasted ASR@k vs. observed ASR@k at a single (step, k) -- the
    same per-behaviour diamond values `plot_asr_decomposition` plots as two separate columns,
    re-paired here by behaviour instead, to isolate whether the forecast gets the *ranking* of
    easy/hard behaviours right, independent of whether the aggregate means happen to be close.
    See `_forecast_vs_observed_points` for the exact forecasted/observed formulas.

    Points above the diagonal: the forecast underestimates that behaviour's ASR@k. Below: it
    overestimates. Colored by rank in `p_hat(step)` (easy -> hard), same convention as
    `plot_asr_decomposition`'s colorbar.

    observed_hypergeometric : if True (default), the observed side is the exact hypergeometric
        tail -- draws without replacement from each behaviour's own finite pool of collected
        samples at `step` (undefined, and so dropped from the scatter, past that pool's own
        size). If False, it's instead the with-replacement `1 - (1-p)**k` plug-in using each
        behaviour's empirical success rate, uncapped by pool size. Same meaning as in
        `plot_asr_decomposition`/`plot_asr_compute_heatmap`.
    title : optional full title override; defaults to a generated
        "step=...  k=...  family=...  fit_cutoff=..." string.

    Behaviours with no defined observed value at `step` are dropped from the scatter.

    Returns fig (== `ax.figure` when `ax` is given).
    """
    from scipy.stats import spearmanr

    sub, row_indices = _converged_rows_for_fit(fit_result, family, fit_cutoff)
    if sub.empty:
        print(f"No converged fits for family={family!r}, cutoff={fit_cutoff!r}; skipping plot.")
        return None

    n_T = len(sub.iloc[0]["predicted_curve"])
    step, k, asr_forecast, asr_observed, colors, valid = _forecast_vs_observed_points(
        sub, row_indices, all_scores, step, k, threshold, n_T, observed_hypergeometric
    )

    created_fig = ax is None
    with plt.rc_context(_SLEEK_RC):
        if created_fig:
            fig, ax = plt.subplots(figsize=(6.5, 6))
        else:
            fig = ax.figure

        _draw_forecast_vs_observed_scatter(ax, asr_forecast, asr_observed, colors, valid, spearmanr)
        ax.set_xlabel("Forecasted ASR@k", fontsize=11)
        ax.set_ylabel("Observed ASR@k", fontsize=11)
        ax.set_title(
            title or f"step={step}  k={k}  family={family}  fit_cutoff={fit_cutoff}",
            fontsize=12,
            loc="left",
            pad=10,
            color="#1a1a1a",
        )

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            label=r"behaviour rank by $\hat{p}(s)$  (easy → hard)",
            fraction=0.046,
            pad=0.02,
            shrink=0.9,
        )
        cbar.outline.set_visible(False)

        if created_fig:
            fig.tight_layout()
    return fig


def plot_asr_forecast_vs_observed_scatter_grid(
    probs, steps, k, observed_hypergeometric=True, title=None
):
    """`plot_asr_forecast_vs_observed_scatter`, one panel per entry in `steps` (typically 4), all
    at the same `k` -- lets forecast-ranking accuracy be compared side by side across attack
    steps. Panels share x/y axes and a single colorbar; only the leftmost panel carries the
    "Observed ASR@k" y-axis label, since it's identical on every panel.

    probs : long-format dataframe, one row per (behaviour, step) that has a value -- columns
        row_id, step, p_forecast, p_true, n_pool, n_succ. See `plot_asr_compute_heatmap` for the
        exact schema. A step in `steps` with no matching rows renders an empty panel (a message
        is printed; the other panels are unaffected).
    steps : sequence of step values, one per panel (in order, left to right).
    k : samples drawn; shared across every panel.
    observed_hypergeometric : see `plot_asr_forecast_vs_observed_scatter` -- applied identically
        to every panel.
    title : optional full suptitle override; defaults to "k=...".

    Returns fig.
    """
    from scipy.stats import spearmanr

    if probs.empty:
        print("`probs` is empty; skipping plot.")
        return None

    k = max(int(k), 1)

    with plt.rc_context(_SLEEK_RC):
        fig, axes = plt.subplots(
            1, len(steps), figsize=(5.2 * len(steps), 6), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, step in zip(axes, steps):
            ax.set_xlabel("Forecasted ASR@k", fontsize=11)
            ax.set_title(f"step={step}", fontsize=12, color="#1a1a1a")

            sub = _probs_at_step(probs, step)
            if sub.empty:
                print(f"No behaviours with data at step={step}; leaving that panel empty.")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.spines[["top", "right"]].set_visible(False)
                continue

            p_hat = np.clip(sub["p_forecast"].to_numpy(dtype=float), 0.0, 1.0)
            asr_forecast = 1 - (1 - p_hat) ** k

            n_pool = sub["n_pool"].to_numpy(dtype=float)
            n_succ = sub["n_succ"].to_numpy(dtype=float)
            p_true = sub["p_true"].to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # 0/empty-pool entries, expected
                asr_observed = _observed_curve_from_probs(
                    n_pool, n_succ, p_true, np.array([k]), observed_hypergeometric
                )[:, 0]

            # Same rank-based coloring as `plot_asr_decomposition`, so points are comparable.
            rank = np.argsort(np.argsort(p_hat))
            norm_rank = rank / max(len(rank) - 1, 1)
            colors = plt.cm.viridis(norm_rank)
            valid = ~np.isnan(asr_observed)

            _draw_forecast_vs_observed_scatter(
                ax, asr_forecast, asr_observed, colors, valid, spearmanr
            )

        axes[0].set_ylabel("Observed ASR@k", fontsize=11)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes,
            label=r"behaviour rank by $\hat{p}(s)$  (easy → hard)",
            fraction=0.02,
            pad=0.02,
            shrink=0.9,
        )
        cbar.outline.set_visible(False)

        fig.suptitle(title or f"k={k}", fontsize=13.5, y=0.98, color="#1a1a1a")
    return fig


def _hypergeom_tail(pool_size, n_succ, k):
    """Vectorized P(>=1 success in k draws without replacement), NaN where `k > pool_size`.

    Scalar `k` only (unlike `_hypergeom_success_curve`, which supports an array of k values via
    an explicit per-element Python loop -- fine once per (behaviour, step), far too slow for a
    (n_bootstrap, n_behaviours, n_steps)-sized call, which is what `compute_asr_at_k`'s
    resampling needs). Uses `scipy.stats.hypergeom.sf`, which is vectorized in C.
    """
    from scipy.stats import hypergeom

    valid = pool_size >= k
    pool_size_safe = np.where(valid, pool_size, k)  # keeps N<=M for scipy; result is masked below
    with np.errstate(invalid="ignore"):
        tail = hypergeom.sf(0, pool_size_safe, n_succ, k)
    return np.where(valid, tail, np.nan)


def compute_asr_at_k(success, k, n_bootstrap=200, ci=0.95, seed=0):
    """Non-cumulative ASR@k(t): probability that k samples drawn without replacement from
    step-t's own pool alone contain >=1 success -- the exact hypergeometric tail (see
    `_hypergeom_tail`), with a nonparametric bootstrap confidence interval on that estimate.

    success : (n_behaviours, n_pool, n_steps) 0/1/NaN array -- pool is every sample recorded
    at that step (e.g. runs x per-run samples, already flattened together).

    Unlike `compute_asr_at_k_cumulative`, there is no "once broken, stays broken" here -- each
    step's value depends only on that step's own pool, not on what happened at earlier steps.
    Because of that, `point`/`per_behaviour` need no simulation at all: they're the exact
    hypergeometric tail computed directly from the observed `pool_size`/`n_succ` counts, not an
    estimate a bootstrap produces. The bootstrap here instead resamples the *data*, to quantify
    genuine finite-sample uncertainty in that estimate -- via the standard equivalence that
    resampling-with-replacement from a fixed 0/1 pool of size N with `n_succ` ones is
    distributed as `Binomial(N, n_succ/N)`, so `n_bootstrap` replicate pools are simulated that
    way and the hypergeometric tail is recomputed on each. This is a different kind of
    "bootstrap" than `compute_asr_at_k_cumulative`'s: that one simulates hypothetical *break
    events* at an already-exactly-known probability, useful there only because the cumulative
    running-max step makes the trajectory-level outcome genuinely path-dependent; there is no
    such compounding here for that style of simulation to be informative about.

    Returns `point`/`lo`/`hi` (n_steps,) -- the aggregate ASR@k(t) estimate (mean over
    behaviours) and its `ci` percentile band from the resampled pools -- and `per_behaviour`
    (n_behaviours, n_steps), the exact per-behaviour estimate (not resampled).
    """
    pool_size = (~np.isnan(success)).sum(axis=1).astype(float)  # (n_behaviours, n_steps)
    n_succ = np.nansum(success, axis=1)
    p = _hypergeom_tail(pool_size, n_succ, k)  # (n_behaviours, n_steps)

    valid = pool_size > 0
    pool_int_safe = np.where(valid, pool_size, 1).astype(int)
    phat_safe = np.where(valid, n_succ / np.maximum(pool_size, 1), 0.0)

    rng = np.random.default_rng(seed)
    n_succ_boot = rng.binomial(pool_int_safe, phat_safe, size=(n_bootstrap, *pool_size.shape))
    p_boot = _hypergeom_tail(pool_size, n_succ_boot, k)  # (n_bootstrap, n_behaviours, n_steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # unobserved (behaviour, step) -> all-NaN
        aggregate_draws = np.nanmean(p_boot, axis=1)  # (n_bootstrap, n_steps)
        point = np.nanmean(p, axis=0)  # (n_steps,)

    lo, hi = np.nanpercentile(aggregate_draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
    return {"point": point, "lo": lo, "hi": hi, "per_behaviour": p}


def compute_asr_at_k_cumulative(success, k, n_bootstrap=200, ci=0.95, seed=0):
    """Bootstrap ASR@k(t): probability that k samples drawn without replacement from the
    step-t pool contain >=1 success, made cumulative over steps (once broken, stays broken).

    success : (n_behaviours, n_pool, n_steps) 0/1/NaN array -- pool is every sample recorded
    at that step (e.g. runs x per-run samples, already flattened together).

    "Did a random k-of-N draw succeed" is exactly a Bernoulli trial with the hypergeometric
    P(>=1 success) (see `_hypergeom_success_curve`), so each bootstrap replicate simulates
    that Bernoulli independently per step -- no resampling of steps or behaviours -- then
    takes a running max over steps and averages over behaviours, matching the
    trajectory-bootstrap in "Sampling-Aware Adversarial Attacks" (arXiv:2412.03556).

    Step 0 is pinned to 0 (nothing is broken before the first optimisation step), so it
    neither counts as a break nor propagates one forward -- this also anchors rate-family
    fits, which assume ASR(0) = 0, at t=0.

    Returns `point`/`lo`/`hi` (n_steps,) -- the aggregate ASR@k(t) estimate and its `ci`
    percentile band -- and `per_behaviour` (n_behaviours, n_steps), the point estimate per
    behaviour (feed this into `fit_and_forecast`; also drawn as the faint background lines).
    `agg_std`/`std` are the bootstrap standard deviation of `aggregate_draws`/`cum` across
    replicates, at the same (n_steps,)/(n_behaviours, n_steps) shapes as `point`/`per_behaviour`.
    """
    pool_size = (~np.isnan(success)).sum(axis=1).astype(float)  # (n_behaviours, n_steps)
    n_succ = np.nansum(success, axis=1)
    p = _hypergeom_success_curve(pool_size, n_succ, np.array([k]))[..., 0]

    ever_valid = np.maximum.accumulate(~np.isnan(p), axis=-1)
    rng = np.random.default_rng(seed)
    draws = rng.random((n_bootstrap, *p.shape)) < p
    draws[..., 0] = False  # ASR(0) = 0: no break before the first step, and none to carry forward
    cum = np.where(ever_valid, np.maximum.accumulate(draws.astype(float), axis=-1), np.nan)
    # cum[..., 0] = 0.0  # ... and reported as 0 even when step 0 is unobserved

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN prefix before first valid step
        per_behaviour = np.nanmean(cum, axis=0)
        aggregate_draws = np.nanmean(cum, axis=1)
        std = np.nanstd(cum, axis=0)
        agg_std = np.nanstd(aggregate_draws, axis=0)

    lo, hi = np.nanpercentile(aggregate_draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
    return {
        "point": np.nanmean(aggregate_draws, axis=0),
        "lo": lo,
        "hi": hi,
        "per_behaviour": per_behaviour,
        "std": std,
        "agg_std": agg_std,
    }


def plot_asr_at_k_fit(models, family1, familyk, cutoff, k, title=None, save=True):
    """Two rows (ASR@1 on top, ASR@k below) x one column per model (left to right), all at a
    single `cutoff`: each panel's empirical curve -- faint per-behaviour background, bootstrap
    CI band, aggregate point estimate, from that model's `boot1`/`bootk`
    (`compute_asr_at_k_cumulative` outputs) -- overlaid with that row's `family1`/`familyk` fit
    at `cutoff`, looked up from that model's `df1`/`dfk` `aggregate` row. A cutoff missing a
    converged fit still shows the empirical curve, just without a fit line.

    `models` is a sequence of `(name, df1, boot1, dfk, bootk)` tuples, one per column, with
    `name` used as that column's header. Meant for comparing models side by side at a single
    cutoff under one shared `title`, in a large enough typeface to stand on its own in print.
    """
    # A deep viridis blue for the empirical/aggregate curve -- black read flat here (this panel
    # already has a gray per-behaviour background, so a black line/CI-band wash doesn't stand
    # out the way it does alone in `plot_asr_decomposition`). Orange stays for the fit overlay,
    # matching those same plots' orange accent markers (the median-ASR star, the `mark_cost`
    # star) -- still one consistent, deliberate pair rather than an arbitrary third color.
    color, fit_color = plt.cm.viridis(0.25), "#e08214"

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, axes = plt.subplots(
            2,
            len(models),
            figsize=(7.0 * len(models), 11),
            sharex="col",
            sharey="row",
            squeeze=False,
        )
        for col, (name, df1, boot1, dfk, bootk) in enumerate(models):
            rows = (("ASR@1", df1, boot1, family1), (f"ASR@{k}", dfk, bootk, familyk))
            for row, (row_name, df, boot, family) in enumerate(rows):
                ax = axes[row, col]
                point, lo, hi = boot["point"], boot["lo"], boot["hi"]
                t = np.arange(len(point))
                valid = ~np.isnan(point)

                for curve in boot["per_behaviour"]:
                    c_valid = ~np.isnan(curve)
                    ax.plot(
                        t[c_valid],
                        curve[c_valid],
                        color="#999999",
                        alpha=0.25,
                        linewidth=1.0,
                        zorder=1,
                    )
                ax.fill_between(
                    t[valid], lo[valid], hi[valid], color=color, alpha=0.25, linewidth=0, zorder=2
                )
                ax.plot(
                    t[valid], point[valid], color=color, linewidth=3, zorder=3, label="empirical"
                )

                match = df[
                    (df["row_id"] == "aggregate")
                    & (df["family"] == family)
                    & (df["cutoff"] == cutoff)
                    & df["converged"]
                ]
                if match.empty:
                    print(
                        f"No converged {family!r} fit at cutoff={cutoff!r} for {name} {row_name}; showing empirical only."
                    )
                else:
                    fit = np.clip(match.iloc[0]["predicted_curve"], 0.0, 1.0)
                    ax.plot(
                        t,
                        fit,
                        color=fit_color,
                        linestyle="--",
                        linewidth=2.5,
                        zorder=4,
                        label=f"{_prettify(family)} fit",
                    )
                if cutoff != "max":
                    ax.axvline(
                        cutoff,
                        color="#555555",
                        linestyle=(0, (4, 3)),
                        linewidth=1.6,
                        alpha=0.6,
                        zorder=1,
                    )

                ax.set_ylim(0, 1)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#d5d5d5", linewidth=1.0, zorder=0)
                ax.tick_params(direction="out", length=5)
                if row == len(rows) - 1 and col == len(models) - 1:
                    ax.legend(
                        frameon=False,
                        fontsize=_BIG_LEGEND_FONT,
                        loc="lower right",
                        handlelength=1.8,
                    )
                if row == 0:
                    ax.set_title(name, fontsize=_BIG_TITLE_FONT)
                if col == 0:
                    ax.set_ylabel(row_name, fontsize=_BIG_FONT)

        for ax in axes.flat:
            ax.label_outer()
        for ax in axes[-1]:
            ax.set_xlabel("Step", fontsize=_BIG_FONT)
        fig.suptitle(
            title,
            fontsize=_BIG_TITLE_FONT,
            wrap=True,
            y=0.999,
        )
        fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.15)
    _save_fig(fig, title, "asr_at_k_fit", save)
    return fig


def plot_asr_at_k_fit_all_families(name, df, boot, cutoff, k, families=None, title=None, save=True):
    """Single-panel variant of `plot_asr_at_k_fit`: for one model at one k and one cutoff, every
    converged family's fit is overlaid in the same unlabeled orange instead of highlighting a
    single family -- useful for eyeballing how much the candidate families agree or disagree.
    Everything else (empirical curve, styling, cutoff guide line) matches `plot_asr_at_k_fit`.

    name : panel title (e.g. a model name).
    df, boot : `fit_and_forecast` results and `compute_asr_at_k_cumulative` output for this k,
        the same shape as one `df1`/`boot1` (or `dfk`/`bootk`) pair in `plot_asr_at_k_fit`.
    cutoff : cutoff to look up each family's fit at.
    families : families to overlay; defaults to every family present in `df` at `cutoff`.
    title : optional suptitle; defaults to None.

    Returns fig.
    """
    # Same viridis-blue/orange convention as `plot_asr_at_k_fit` -- see its comment.
    color, fit_color = plt.cm.viridis(0.25), "#e08214"
    if families is None:
        families = sorted(df.loc[df["cutoff"] == cutoff, "family"].astype(str).unique())

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, ax = plt.subplots(figsize=(7.0, 6))

        point, lo, hi = boot["point"], boot["lo"], boot["hi"]
        t = np.arange(len(point))
        valid = ~np.isnan(point)

        # for curve in boot["per_behaviour"]:
        #     c_valid = ~np.isnan(curve)
        #     ax.plot(
        #         t[c_valid], curve[c_valid], color="#999999", alpha=0.25, linewidth=1.0, zorder=1
        #     )
        ax.fill_between(
            t[valid], lo[valid], hi[valid], color=color, alpha=0.25, linewidth=0, zorder=2
        )
        ax.plot(t[valid], point[valid], color=color, linewidth=3, zorder=3, label="empirical")

        any_fit = False
        for family in families:
            match = df[
                (df["row_id"] == "aggregate")
                & (df["family"] == family)
                & (df["cutoff"] == cutoff)
                & df["converged"]
            ]
            if match.empty:
                print(f"No converged {family!r} fit at cutoff={cutoff!r}; skipping.")
                continue
            fit = np.clip(match.iloc[0]["predicted_curve"], 0.0, 1.0)
            ax.plot(
                t,
                fit,
                color=fit_color,
                linestyle="--",
                linewidth=2.5,
                zorder=4,
                label="fit" if not any_fit else None,
            )
            any_fit = True

        if cutoff != "max":
            ax.axvline(
                cutoff, color="#555555", linestyle=(0, (4, 3)), linewidth=1.6, alpha=0.6, zorder=1
            )

        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d5d5d5", linewidth=1.0, zorder=0)
        ax.tick_params(direction="out", length=5)
        # ax.legend(frameon=False, fontsize=_BIG_LEGEND_FONT, loc="lower right", handlelength=1.8)
        # ax.set_title(name, fontsize=_BIG_TITLE_FONT)
        # ax.set_ylabel(f"ASR@k", fontsize=_BIG_FONT)
        ax.set_xlabel("Step", fontsize=_BIG_FONT)

        fig.suptitle(title, fontsize=_BIG_TITLE_FONT, wrap=True, y=0.999)
        fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.15)
    _save_fig(fig, title, "asr_at_k_fit_all_families", save)
    return fig


def plot_asr_mean_prob_vs_observed(models, k, title=None, save=True):
    """One panel per model: Step vs ASR@k, "true observed" (`1-(1-p)**k` per behaviour, then
    averaged) vs "mean-probability forecast" (averaged first, then plugged in once) -- Jensen's
    inequality means the forecast never exceeds the observed value.

    models : sequence of `(name, success_prob)` pairs; success_prob is (n_behaviours, n_steps).
    """
    # Same viridis-blue/orange convention as `plot_asr_at_k_fit` -- see its comment.
    color, forecast_color = plt.cm.viridis(0.25), "#e08214"

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT}):
        fig, axes = plt.subplots(
            1, len(models), figsize=(7.5 * len(models), 6), sharey=True, squeeze=False
        )
        for ax, (name, success_prob) in zip(axes[0], models, strict=True):
            success_prob = np.asarray(success_prob)
            t = np.arange(success_prob.shape[1])
            # Steps where every behaviour is NaN (no data recorded yet) would otherwise break
            # the line there; skip them like `_plot_overlaid_lines` does, so the line connects
            # straight across the gap to the next step that does have data.
            has_data = ~np.all(np.isnan(success_prob), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # steps with all-NaN behaviours
                mean_p = np.nanmean(success_prob, axis=0)
                observed_asr = np.nanmean(1 - (1 - success_prob) ** k, axis=0)

            ax.plot(
                t[has_data],
                observed_asr[has_data],
                color=color,
                linewidth=3,
                zorder=3,
                label="true observed",
            )
            ax.plot(
                t[has_data],
                (1 - (1 - mean_p) ** k)[has_data],
                color=forecast_color,
                linestyle="--",
                linewidth=2.5,
                zorder=4,
                label="mean-probability forecast",
            )

            ax.set_ylim(0, 1)
            ax.set_title(name, fontsize=_BIG_TITLE_FONT)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#d5d5d5", linewidth=1.0, zorder=0)
            ax.tick_params(direction="out", length=5)
            ax.set_xlabel("Step", fontsize=_BIG_FONT)

        axes[0, 0].set_ylabel(f"ASR@{k}", fontsize=_BIG_FONT)
        axes[0, -1].legend(
            frameon=False, fontsize=_BIG_LEGEND_FONT, loc="lower right", handlelength=1.8
        )

        fig.suptitle(title, fontsize=_BIG_TITLE_FONT)
        fig.tight_layout(pad=1.4, w_pad=0.6)
        plt.show()  # so it renders even when the call is assigned, e.g. `fig, axes = plot_...(...)`
    _save_fig(fig, title, "asr_mean_prob_vs_observed", save)
    return fig, axes


def _cutoff_order(cutoffs):
    """Sort cutoffs numerically, with 'max' (if present) split out separately."""

    def key(v):
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    is_max = lambda v: isinstance(v, str) and v.strip().lower() == "max"
    order = sorted(cutoffs, key=key)
    regular = [c for c in order if not is_max(c)]
    has_max = any(is_max(c) for c in order)
    return regular, has_max


def _canon_cutoff(v):
    """Collapse mixed representations of the same cutoff (10 vs '10') to one
    value, and normalize 'max' spelling/whitespace/case."""
    if isinstance(v, str) and v.strip().lower() == "max":
        return "max"
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return str(v)


def _all_converged(s):
    s = s.dropna()
    return bool(s.all()) if len(s) else True  # unknown -> don't flag as non-converged


def _plot_panel(
    ax,
    df,
    metric,
    max_metric,
    families,
    colors,
    lower_is_better,
    family_col,
    cutoff_col,
    converged_col,
    highlight_best,
    x_step=1.0,
):
    df = df.copy()
    df[family_col] = df[family_col].astype(str)
    n_before = len(df)
    df = df[df[cutoff_col].notna()]  # drop rows with no recorded cutoff
    if len(df) < n_before:
        print(f"Warning: dropped {n_before - len(df)} row(s) with missing '{cutoff_col}'.")
    df[cutoff_col] = df[cutoff_col].apply(_canon_cutoff)
    has_conv = converged_col in df.columns

    dup_counts = df.groupby([family_col, cutoff_col]).size()
    n_dup = (dup_counts > 1).sum()
    if n_dup:
        example = dup_counts[dup_counts > 1].index[0]
        print(
            f"Warning: {n_dup} (family, cutoff) combination(s) have more than one row "
            f"in this dataframe (e.g. {example} has {dup_counts[example]} rows). Only "
            f"converged rows are averaged together — check this dataframe is already "
            f"filtered to a single attack/model/k, as duplicates usually mean it isn't."
        )

    # only converged rows contribute to the plotted value, so a non-converged
    # attempt can never silently drag a real number in via averaging; if a
    # (family, cutoff) has zero converged rows the point is skipped entirely
    metric_rows = df[df[converged_col] == True] if has_conv else df

    regular, has_max = _cutoff_order(df[cutoff_col].unique())
    xpos = [i * x_step for i in range(len(regular))]
    tick_pos, tick_labels = list(xpos), [str(c) for c in regular]

    # connected line over the regular (non-"max") cutoffs
    pivot = (
        metric_rows[metric_rows[cutoff_col].isin(regular)]
        .pivot_table(index=family_col, columns=cutoff_col, values=metric, aggfunc="mean")
        .reindex(index=families, columns=regular)
    )
    conv = (
        df[df[cutoff_col].isin(regular)]
        .pivot_table(
            index=family_col, columns=cutoff_col, values=converged_col, aggfunc=_all_converged
        )
        .reindex(index=families, columns=regular)
        if has_conv
        else None
    )

    for f in families:
        y = pivot.loc[f]
        ax.plot(
            xpos,
            y,
            color=colors[f],
            linestyle=family_linestyle(f),
            linewidth=5.0,
            alpha=0.9,
            zorder=3,
            solid_capstyle="round",
            dash_capstyle="round",
        )
        for i, c in enumerate(regular):
            if pd.isna(y[c]):
                continue  # no converged attempt for this (family, cutoff) -> leave a gap
            converged = conv.loc[f, c] if conv is not None else True
            ax.scatter(
                [xpos[i]],
                [y[c]],
                marker="o" if converged else "^",
                color=colors[f],
                s=90,
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
                clip_on=False,
            )

    if highlight_best:
        valid_cols = pivot.columns[pivot.notna().any(axis=0)]
        if len(valid_cols):
            leaders = (
                pivot[valid_cols].idxmin(axis=0)
                if lower_is_better
                else pivot[valid_cols].idxmax(axis=0)
            )
            for i, c in enumerate(regular):
                if c in leaders.index:
                    f = leaders[c]
                    converged = conv.loc[f, c] if conv is not None else True
                    ax.scatter(
                        [xpos[i]],
                        [pivot.loc[f, c]],
                        marker="o" if converged else "^",
                        s=95,
                        facecolor=colors[f],
                        edgecolor="#1a1a1a",
                        linewidth=1.3,
                        zorder=6,
                        clip_on=False,
                    )

    # star at "max", from a separate column (e.g. in-sample metric, since holdout
    # metrics are undefined once all data is used for fitting), connected back to
    # each family's last regular point with a dashed segment (dashed = metric switch).
    # Non-converged points are drawn hollow rather than switched to a triangle, so
    # the star shape still marks "this is the max column" unambiguously.
    if has_max:
        max_x = len(regular) * x_step
        tick_pos.append(max_x)
        tick_labels.append("max")
        if max_metric is not None:
            max_rows = df[df[cutoff_col] == "max"]
            max_rows_metric = max_rows[max_rows[converged_col] == True] if has_conv else max_rows
            star_vals = max_rows_metric.groupby(family_col)[max_metric].mean().reindex(families)
            star_conv = (
                max_rows.groupby(family_col)[converged_col].apply(_all_converged).reindex(families)
                if has_conv
                else None
            )
            for f in families:
                v = star_vals.get(f, np.nan)
                if pd.isna(v):
                    continue
                if regular and pd.notna(pivot.loc[f, regular[-1]]):
                    ax.plot(
                        [xpos[-1], max_x],
                        [pivot.loc[f, regular[-1]], v],
                        color=colors[f],
                        linewidth=2.8,
                        linestyle=(0, (5, 2)),
                        alpha=0.9,
                        zorder=3,
                    )
                converged = star_conv.get(f, True) if star_conv is not None else True
                face = colors[f] if converged else "none"
                ax.scatter(
                    [max_x],
                    [v],
                    marker="*",
                    s=150,
                    facecolor=face,
                    edgecolor=colors[f],
                    linewidth=1.3,
                    zorder=5,
                    clip_on=False,
                )
            if highlight_best and star_vals.notna().any():
                f_lead = star_vals.idxmin() if lower_is_better else star_vals.idxmax()
                lead_converged = star_conv.get(f_lead, True) if star_conv is not None else True
                face = colors[f_lead] if lead_converged else "none"
                ax.scatter(
                    [max_x],
                    [star_vals[f_lead]],
                    marker="*",
                    s=360,
                    facecolor=face,
                    edgecolor="#1a1a1a",
                    linewidth=1.5,
                    zorder=7,
                    clip_on=False,
                )

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    # Compact x-range: first cutoff sits on the y-axis, right edge stops just past the
    # last tick ("max" when present) instead of the wide symmetric margin.
    ax.set_xlim(-0.08 * x_step, (max(tick_pos) if tick_pos else 1) + 0.3 * x_step)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.tick_params(direction="out", length=6, labelsize=_BIG_FONT * 3)


def plot_family_comparison_grid(
    dfs,
    metric,
    title=None,
    max_metric=None,
    model_names=("Llama 3.1", "Gemma 3"),
    k_labels=("k=1", "k=100"),
    lower_is_better=True,
    family_col="family",
    cutoff_col="cutoff",
    converged_col="converged",
    highlight_best=True,
    show_ylabel=True,
    x_step=1.0,
    figsize=(44, 25.2),
    save=True,
):
    """
    2x2 grid (single attack, single metric): rows = sampling volume k, columns = model.
    Each family keeps its fixed color from `family_color` -- the same hue it has in
    every other plot in this module -- shared across panels via a single figure legend
    below the grid (rather than end-of-line labels, which overlap with many families).
    The k tag for each row rides on that row's y-axis label; panels are drawn with a
    tight x-range (first cutoff on the y-axis, right edge just past "max") and cutoffs
    packed closer together via `x_step`.

    dfs : nested list [[df_k0_model0, df_k0_model1], [df_k1_model0, df_k1_model1]].
    metric : column plotted as a connected line across non-"max" cutoffs.
    max_metric : column plotted as a star at the "max" cutoff, connected back to
        each family's line with a dashed segment (e.g. an in-sample metric, since
        holdout metrics are undefined once all data is used for fitting).
        Pass None to skip the "max" point's value (the "max" tick still appears
        on the x-axis if any row has that cutoff).
    lower_is_better : bool or list of 2 bools [for metric, for max_metric].
    converged_col : boolean column; non-converged points are drawn as a triangle
        (filled star -> hollow star, for the "max" column) instead of a circle.
    highlight_best : if True, outline the best family's marker at each cutoff
        (and at "max") with a larger, dark-edged marker.
    show_ylabel : if False, omit the row's "k=.../metric" y-axis label entirely
        (e.g. when the grid is placed next to other panels that already carry it).
    x_step : spacing between adjacent cutoff ticks, in axes units. Defaults to 1.0
        (one-unit-per-cutoff) so the large-print tick labels ("200", "400", ..., "max")
        have room to sit without crowding; values below 1 pack the x-axis tighter, at
        the cost of that clearance.

    Styling follows the sleek house style shared with `plot_asr_at_k_fit`,
    `plot_asr_compute_heatmap` and `plot_asr_decomposition`: `_SLEEK_LATEX_RC` fonts
    (Computer Modern, at the thesis's large-print-figure scale), top/right spines
    dropped, a faint y-grid, and underscores stripped from labels.
    """
    if isinstance(lower_is_better, bool):
        lower_is_better = [lower_is_better, lower_is_better]
    families = sorted(set().union(*[set(d[family_col].astype(str)) for row in dfs for d in row]))
    colors = {
        f: family_color(f) for f in families
    }  # fixed per family, shared with every other plot
    model_names = list(model_names) if model_names is not None else ["Model A", "Model B"]
    k_labels = list(k_labels) if k_labels is not None else ["k (row 1)", "k (row 2)"]

    with plt.rc_context({**_SLEEK_LATEX_RC, "font.size": _BIG_FONT * 3}):
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharey="row")
        for r in range(2):
            for c in range(2):
                _plot_panel(
                    axes[r][c],
                    dfs[r][c],
                    metric,
                    max_metric,
                    families,
                    colors,
                    lower_is_better[0],
                    family_col,
                    cutoff_col,
                    converged_col,
                    highlight_best,
                    x_step=x_step,
                )
                if r == 0:
                    axes[r][c].set_title(
                        _prettify(model_names[c]), fontsize=_BIG_TITLE_FONT * 3, loc="left", pad=6
                    )
            if show_ylabel:
                # Row identity (k) and metric share the y-axis label -- keeps the k tag right
                # next to its panel (no far-left gutter) and echoes `plot_asr_at_k_fit`.
                axes[r][0].set_ylabel(
                    f"{_prettify(k_labels[r])}\n{_prettify(metric)}",
                    fontsize=_BIG_FONT * 3,
                    linespacing=1.4,
                )
        for ax in axes[-1]:
            ax.set_xlabel(_prettify(cutoff_col), fontsize=_BIG_FONT * 3)

        handles = [
            mlines.Line2D(
                [],
                [],
                color=colors[f],
                linestyle=family_linestyle(f),
                marker="o",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=5.0,
                label=_prettify(f),
            )
            for f in families
        ]
        ncols = min(len(handles), 3)
        n_rows = -(-len(handles) // ncols)
        bottom = 0.08 + 0.05 * n_rows  # room for the wrapped legend rows
        left = 0.095 if show_ylabel else 0.065
        fig.subplots_adjust(
            left=left, right=0.965, top=0.89, bottom=bottom, hspace=0.14, wspace=0.03
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            ncols=ncols,
            fontsize=_BIG_LEGEND_FONT * 3,
            handlelength=3,
            columnspacing=1.4,
            borderaxespad=0.0,
        )

        # title = f"Family comparison — {_prettify(metric)}"
        # if max_metric:
        # cmr10 has no Unicode star glyph -- "$\\star$" renders via mathtext (fontset "cm")
        # instead, which does have one, rather than falling back to a missing-glyph box.
        # title += f"   ($\\star$ = {_prettify(max_metric)} at cutoff = max)"
        fig.suptitle(title, fontsize=_BIG_TITLE_FONT * 3, y=0.975)
    _save_fig(fig, title, "family_comparison_grid", save)
    return fig, axes
