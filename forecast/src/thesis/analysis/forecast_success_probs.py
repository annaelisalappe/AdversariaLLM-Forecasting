from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

__all__ = [
    "SuccessProbFit",
    "asr_endpoint_error",
    "build_asr_probs_from_fit",
    "fit_success_probs",
    "forecast_asr_at_k",
    "forecast_asr_at_k_bootstrap",
    "pinball_score",
    "plot_asr_at_k_forecast",
    "plot_success_prob_forecast",
]

logger = logging.getLogger(__name__)

_LOGIT_PMAX_PRIOR_MEAN = -1.386  # logit(0.2) -- group-level ceiling center


@dataclass
class SuccessProbFit:
    """Posterior draws from one hierarchical fit (one `cutoff`)."""

    intercepts: np.ndarray  # (n_draws, n_behaviours) -- a_i
    slopes: np.ndarray  # (n_draws, n_behaviours) -- b_i
    p_max: np.ndarray  # (n_draws, n_behaviours) -- per-behaviour saturation ceiling
    concentration: np.ndarray | None  # (n_draws,) -- pooled beta-binomial concentration, or None if binomial
    cutoff: int | str
    t_values: np.ndarray  # (n_steps,) step number per column
    s_ref: float
    category_names: list[str] = field(repr=False)
    behaviour_category: np.ndarray = field(repr=False)  # (n_behaviours,) index into category_names
    use_beta_binomial: bool = True  # which likelihood produced this fit

    def predict(self, s: float | np.ndarray, q: float) -> np.ndarray:
        """Posterior quantile `q` of the success probability at step(s) `s`.

        Returns shape `(n_behaviours,)` for scalar `s`, `(n_behaviours, len(s))` for array `s`.
        """
        s = np.asarray(s, dtype=float)
        scalar_input = s.ndim == 0
        log_s = np.atleast_1d(np.log10(s) - np.log10(self.s_ref))
        logit = self.intercepts[:, :, None] + self.slopes[:, :, None] * log_s  # (draws, beh, len(s))
        p = self.p_max[:, :, None] / (1.0 + np.exp(-logit))
        out = np.quantile(p, q, axis=0)  # (n_behaviours, len(s))
        return out[:, 0] if scalar_input else out


def _model(
    beh_cat_id,
    beh_id,
    log_s,
    n_trials,
    n_success,
    n_categories,
    n_behaviours,
    use_beta_binomial: bool = True,
):
    mu_a = numpyro.sample("mu_a", dist.Normal(0.5, 0.5))
    mu_b = numpyro.sample("mu_b", dist.Normal(0.5, 1.0))  # favours a rise, doesn't forbid a decline

    sigma_a_cat = numpyro.sample("sigma_a_cat", dist.HalfNormal(3.0))  # intercepts vary freely
    sigma_b_cat = numpyro.sample("sigma_b_cat", dist.HalfNormal(0.15))  # tightly pooled slope
    sigma_a_beh = numpyro.sample("sigma_a_beh", dist.HalfNormal(3.0))
    sigma_b_beh = numpyro.sample("sigma_b_beh", dist.HalfNormal(0.4))  # modestly pooled slope

    # --- ceiling p_max: same style of hierarchy as a_i. Wide, data-adaptive hyperpriors on the
    # sigmas mean pooling strength is *learned*, not fixed: behaviours with a clearly observed
    # plateau can depart from the group mean, while sparse behaviours still shrink toward it.
    mu_pmax = numpyro.sample("mu_pmax", dist.Normal(_LOGIT_PMAX_PRIOR_MEAN, 1.0))
    sigma_pmax_cat = numpyro.sample("sigma_pmax_cat", dist.HalfNormal(3.0))
    sigma_pmax_beh = numpyro.sample("sigma_pmax_beh", dist.HalfNormal(3.0))

    if use_beta_binomial:
        phi = numpyro.sample("phi", dist.HalfNormal(0.05))
        concentration = numpyro.deterministic("concentration", 1.0 / jnp.clip(phi, 1e-6))

    with numpyro.plate("categories", n_categories):
        a_cat_raw = numpyro.sample("a_cat_raw", dist.Normal(0.0, 1.0))
        b_cat_raw = numpyro.sample("b_cat_raw", dist.Normal(0.0, 1.0))
        pmax_cat_raw = numpyro.sample("pmax_cat_raw", dist.Normal(0.0, 1.0))
    a_cat = mu_a + sigma_a_cat * a_cat_raw
    b_cat = mu_b + sigma_b_cat * b_cat_raw
    pmax_cat = mu_pmax + sigma_pmax_cat * pmax_cat_raw

    with numpyro.plate("behaviours", n_behaviours):
        a_beh_raw = numpyro.sample("a_beh_raw", dist.Normal(0.0, 1.0))
        b_beh_raw = numpyro.sample("b_beh_raw", dist.Normal(0.0, 1.0))
        pmax_beh_raw = numpyro.sample("pmax_beh_raw", dist.Normal(0.0, 1.0))
    a = numpyro.deterministic("a", a_cat[beh_cat_id] + sigma_a_beh * a_beh_raw)
    b = numpyro.deterministic("b", b_cat[beh_cat_id] + sigma_b_beh * b_beh_raw)
    p_max = numpyro.deterministic(
        "p_max", jax.nn.sigmoid(pmax_cat[beh_cat_id] + sigma_pmax_beh * pmax_beh_raw)
    )

    p = numpyro.deterministic("p", p_max[beh_id] * jax.nn.sigmoid(a[beh_id] + b[beh_id] * log_s))

    with numpyro.plate("obs", len(n_trials)):
        if use_beta_binomial:
            numpyro.sample(
                "y",
                dist.BetaBinomial(p * concentration, (1.0 - p) * concentration, total_count=n_trials),
                obs=n_success,
            )
        else:
            numpyro.sample("y", dist.Binomial(total_count=n_trials, probs=p), obs=n_success)


def _resolve_categories(categories: Mapping[str, Sequence[int]], n_behaviours: int) -> np.ndarray:
    behaviour_category = np.full(n_behaviours, -1, dtype=int)
    for cat_id, idxs in enumerate(categories.values()):
        behaviour_category[list(idxs)] = cat_id
    if np.any(behaviour_category < 0):
        raise ValueError("`categories` must cover every behaviour index exactly once.")
    return behaviour_category


def _counts(success: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (behaviour, step) pairs in `cols` with >=1 observed sample to long format."""
    n_trials_all = np.sum(~np.isnan(success[:, :, cols]), axis=1)  # (n_behaviours, len(cols))
    with np.errstate(invalid="ignore"):
        n_success_all = np.nansum(success[:, :, cols], axis=1)
    beh_id, step_idx = np.nonzero(n_trials_all > 0)
    return beh_id, cols[step_idx], n_trials_all[beh_id, step_idx], n_success_all[beh_id, step_idx]


def fit_success_probs(
    success: np.ndarray,
    categories: Mapping[str, Sequence[int]],
    cutoffs: Sequence[int | str] = ("max",),
    t_values: np.ndarray | None = None,
    s_ref: float = 100.0,
    use_beta_binomial: bool = True,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    random_seed: int = 0,
    progress_bar: bool = True,
) -> dict[int | str, SuccessProbFit]:
    """Fit the hierarchical model jointly on every behaviour, once per cutoff.

    Args:
        success: `(n_behaviours, n_samples, n_steps)` 0/1/NaN array.
        categories: category name -> behaviour indices (into axis 0), covering every index
            exactly once -- same convention as `categories_to_idx` elsewhere in this package.
        cutoffs: column indices (fit `t <= cutoff`, hold out `t > cutoff`) and/or `"max"`
            (fit every observed column, no holdout).
        t_values: step number per column of `success`. Must be strictly positive (the model
            regresses on log10(step)). Defaults to `arange(1, n_steps + 1)`.
        s_ref: reference step at which `a_i` is the log-odds of success.
        use_beta_binomial: if True (default), use a beta-binomial likelihood with a pooled
            overdispersion parameter (`phi`/`concentration`). If False, use a plain binomial
            likelihood instead -- the `phi`/`concentration` sites are simply not sampled, so
            this is a one-flag round trip back to a true binomial process.

    Returns:
        `{cutoff: SuccessProbFit}`, one joint fit per requested cutoff.
    """
    success = np.asarray(success, dtype=float)
    if success.ndim != 3:
        raise ValueError(f"`success` must be 3D (n_behaviours, n_samples, n_steps); got shape {success.shape}")
    n_behaviours, _, n_steps = success.shape

    if t_values is None:
        t_values = np.arange(1, n_steps + 1, dtype=float)
    else:
        t_values = np.asarray(t_values, dtype=float)
        if t_values.shape != (n_steps,):
            raise ValueError(f"t_values must have shape ({n_steps},); got {t_values.shape}")
    if np.any(t_values <= 0):
        raise ValueError("t_values must be strictly positive: log10(step) is undefined at/below 0.")

    behaviour_category = _resolve_categories(categories, n_behaviours)
    category_names = list(categories.keys())
    obs_cols = np.flatnonzero(np.any(~np.isnan(success), axis=(0, 1)))
    if len(obs_cols) == 0:
        raise ValueError("`success` contains no observed (non-NaN) values.")

    fits: dict[int | str, SuccessProbFit] = {}
    for cutoff in cutoffs:
        if cutoff == "max":
            fit_cols = obs_cols
        else:
            fit_cols = obs_cols[obs_cols <= cutoff]
            if len(fit_cols) == 0 or len(obs_cols[obs_cols > cutoff]) == 0:
                raise ValueError(f"cutoff={cutoff} leaves no observed fit or no held-out columns.")

        beh_id, cols, n_trials, n_succ = _counts(success, fit_cols)
        log_s = np.log10(t_values[cols]) - np.log10(s_ref)

        logger.info(
            "fit_success_probs: cutoff=%r, %d behaviours x %d categories, %d observations, "
            "likelihood=%s",
            cutoff, n_behaviours, len(category_names), len(beh_id),
            "beta-binomial" if use_beta_binomial else "binomial",
        )
        mcmc = MCMC(NUTS(_model), num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=progress_bar)
        mcmc.run(
            jax.random.PRNGKey(random_seed),
            beh_cat_id=jnp.asarray(behaviour_category),
            beh_id=jnp.asarray(beh_id),
            log_s=jnp.asarray(log_s),
            n_trials=jnp.asarray(n_trials),
            n_success=jnp.asarray(n_succ),
            n_categories=len(category_names),
            n_behaviours=n_behaviours,
            use_beta_binomial=use_beta_binomial,
        )
        samples = mcmc.get_samples()
        fits[cutoff] = SuccessProbFit(
            intercepts=np.asarray(samples["a"]),
            slopes=np.asarray(samples["b"]),
            p_max=np.asarray(samples["p_max"]),
            concentration=np.asarray(samples["concentration"]) if use_beta_binomial else None,
            cutoff=cutoff,
            t_values=t_values,
            s_ref=s_ref,
            category_names=category_names,
            behaviour_category=behaviour_category,
            use_beta_binomial=use_beta_binomial,
        )
    return fits


def pinball_score(fit: SuccessProbFit, success: np.ndarray, q: float) -> dict:
    """Pinball loss at quantile `q` on `fit`'s held-out steps, plus empirical coverage.

    Reports the mean pinball loss overall and restricted to behaviours that were all-zero
    inside the fit window (the case the pooled slope is meant to rescue), and the fraction of
    held-out points at or under the forecast (should land near `q` if well calibrated).
    """
    if fit.cutoff == "max":
        raise ValueError("no held-out steps for cutoff='max'.")
    success = np.asarray(success, dtype=float)
    obs_cols = np.flatnonzero(np.any(~np.isnan(success), axis=(0, 1)))
    hold_cols = obs_cols[obs_cols > fit.cutoff]
    fit_cols = obs_cols[obs_cols <= fit.cutoff]
    if len(hold_cols) == 0:
        raise ValueError(f"no held-out observed columns beyond cutoff={fit.cutoff}.")

    beh_id, cols, n_trials, n_succ = _counts(success, hold_cols)
    y = n_succ / n_trials
    log_s = np.log10(fit.t_values[cols]) - np.log10(fit.s_ref)
    logit = fit.intercepts[:, beh_id] + fit.slopes[:, beh_id] * log_s  # (draws, n_obs)
    p = fit.p_max[:, beh_id] / (1.0 + np.exp(-logit))  # (draws, n_obs)
    p_hat = np.quantile(p, q, axis=0)  # (n_obs,)

    def pinball(y, yhat):
        diff = y - yhat
        return np.mean(np.maximum(q * diff, (q - 1.0) * diff))

    # behaviours with zero total successes anywhere in the fit window
    fbeh_id, _, _, fsucc = _counts(success, fit_cols)
    fit_success_by_beh = np.zeros(success.shape[0])
    np.add.at(fit_success_by_beh, fbeh_id, fsucc)
    all_zero_beh = set(np.flatnonzero(fit_success_by_beh == 0))

    mask_all_zero = np.isin(beh_id, list(all_zero_beh))
    return {
        "pinball_overall": pinball(y, p_hat),
        "pinball_all_zero": pinball(y[mask_all_zero], p_hat[mask_all_zero]) if mask_all_zero.any() else None,
        "coverage": float(np.mean(y <= p_hat)),
        "n_holdout": len(y),
        "n_all_zero_behaviours": len(all_zero_beh),
    }


def forecast_asr_at_k(fit: SuccessProbFit, k: int, q: float = 0.5) -> np.ndarray:
    """Posterior quantile `q` of the cumulative ASR@k(s) implied by `fit`, for every step in
    `fit.t_values` -- the model-based counterpart of `compute_asr_at_k_cumulative`'s empirical
    `per_behaviour` curve (same "once broken, stays broken", independent-per-step-draws
    definition). The quantile is taken *after* compounding each posterior draw's own curve
    across steps, not before -- quantile and cumulative product don't commute.

    Returns `(n_behaviours, n_steps)`, aligned to `fit.t_values` by column position.
    """
    log_s = np.log10(fit.t_values) - np.log10(fit.s_ref)
    logit = fit.intercepts[:, :, None] + fit.slopes[:, :, None] * log_s  # (draws, beh, steps)
    p = fit.p_max[:, :, None] / (1.0 + np.exp(-logit))
    break_prob = 1.0 - (1.0 - p) ** k
    cum = 1.0 - np.cumprod(1.0 - break_prob, axis=2)  # (draws, beh, steps)
    return np.quantile(cum, q, axis=0)


def build_asr_probs_from_fit(
    fit: SuccessProbFit, all_scores: np.ndarray, threshold: float = 0.5,
    pool_runs: bool = True, q: float = 0.5, n_endpoint: int = 5,
):
    """Same `(row_id, step, p_forecast, p_true, n_pool, n_succ)` schema as
    `plot_forecasts.build_asr_probs`, but `p_forecast` comes from this hierarchical fit's
    posterior quantile `q` at each column's real step (`fit.t_values`) instead of a
    `fit_and_forecast` `predicted_curve` -- a drop-in for `plot_asr_compute_heatmap`,
    `plot_asr_decomposition`, and `plot_asr_forecast_vs_observed_scatter_grid`.

    `all_scores` is the raw judge-score array (unthresholded), same as `build_asr_probs`
    expects -- `step` here indexes its column axis directly (0-based), not `fit.t_values`.

    Returns `(probs, rmse, holdout_endpoint_error)`:
      - `probs` is the dataframe above.
      - `rmse` is the RMSE between `p_forecast` and `p_true` at every (behaviour, step) with an
        observed `p_true`, computed per behaviour and then averaged across behaviours (so a
        behaviour with more observed steps doesn't dominate the score just for having more of
        them).
      - `holdout_endpoint_error` is `forecast.py`'s `holdout_endpoint_error` metric, adapted
        the same way: mean absolute error between `p_forecast` and `p_true` over the last
        `n_endpoint` observed columns beyond `fit.cutoff`, per behaviour then averaged across
        behaviours. `None` for a `cutoff="max"` fit (no held-out columns) or if none of the
        columns beyond the cutoff have an observed `p_true`.
    """
    import pandas as pd

    from thesis.analysis.plot_forecasts import _observed_success_pool

    n_behaviours = len(fit.behaviour_category)
    n_T = len(fit.t_values)
    n_steps_available = all_scores.shape[3]
    row_indices = list(range(n_behaviours))

    p_forecast = np.clip(fit.predict(fit.t_values, q), 0.0, 1.0)  # (n_behaviours, n_T)

    p_true = np.full((n_behaviours, n_T), np.nan)
    n_pool_arr = np.full((n_behaviours, n_T), np.nan)
    n_succ_arr = np.full((n_behaviours, n_T), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 0/empty-pool steps, expected
        for step in range(min(n_T, n_steps_available)):
            n_pool, n_succ = _observed_success_pool(all_scores, row_indices, step, threshold, pool_runs=pool_runs)
            valid = n_pool > 0
            p_true[valid, step] = n_succ[valid] / n_pool[valid]
            n_pool_arr[valid, step] = n_pool[valid]
            n_succ_arr[valid, step] = n_succ[valid]

    row_id_col = np.repeat([str(r) for r in row_indices], n_T)
    step_col = np.tile(np.arange(n_T), n_behaviours)
    probs = pd.DataFrame({
        "row_id": row_id_col,
        "step": step_col,
        "p_forecast": p_forecast.ravel(),
        "p_true": p_true.ravel(),
        "n_pool": n_pool_arr.ravel(),
        "n_succ": n_succ_arr.ravel(),
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a behaviour with no observed steps
        rmse_per_behaviour = np.sqrt(np.nanmean((p_forecast - p_true) ** 2, axis=1))
        rmse = float(np.nanmean(rmse_per_behaviour))

    holdout_endpoint_error = None
    if fit.cutoff != "max":
        hold_cols = np.flatnonzero((np.arange(n_T) > fit.cutoff) & np.any(~np.isnan(p_true), axis=0))
        cols = hold_cols[-n_endpoint:]
        if len(cols) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a behaviour missing all of `cols`
                err = np.abs(p_forecast[:, cols] - p_true[:, cols])
                holdout_endpoint_error = float(np.nanmean(np.nanmean(err, axis=1)))

    return probs, rmse, holdout_endpoint_error


def plot_success_prob_forecast_with_residuals(
    fit: SuccessProbFit, success: np.ndarray, categories: Mapping[str, Sequence[int]],
    q_upper: float = 0.9, y_label: str = "Success probability", title: str | None = None,
):
    from thesis.analysis.analyse import _category_colors
 
    success = np.asarray(success, dtype=float)
    with np.errstate(invalid="ignore"):
        observed = np.nanmean(success, axis=1)  # (n_behaviours, n_steps)
    t = fit.t_values
    median = fit.predict(t, 0.5)
    upper = fit.predict(t, q_upper)
    residual = observed - median  # (n_behaviours, n_steps), NaN where unobserved
 
    _, legend = _category_colors(success.shape[0], categories)
    cat_colors = dict(legend)
 
    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 2, figsize=(14, 3.5 * n_cats), sharex=True, squeeze=False)
 
    resid_max = np.nanmax(np.abs(residual)) if np.any(~np.isnan(residual)) else 0.1
    resid_lim = max(0.05, resid_max * 1.1)  # small floor so a near-zero-residual fit isn't over-zoomed
 
    for row, (cat_name, idxs) in enumerate(categories.items()):
        color = cat_colors[cat_name]
        ax_resid, ax_curve = axes[row, 0], axes[row, 1]
 
        # --- residual panel (left) ---
        for i in idxs:
            has = ~np.isnan(residual[i])
            ax_resid.plot(t[has], residual[i][has], color=color, alpha=0.25, linewidth=0.7)
        mean_resid = np.nanmean(residual[idxs], axis=0)
        ax_resid.plot(t, mean_resid, color=color, linewidth=2.0)
        ax_resid.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
        if fit.cutoff != "max":
            ax_resid.axvspan(0.0, t[fit.cutoff], color="k", alpha=0.06)
        ax_resid.set_xlim(left=0.0)
        ax_resid.set_ylim(-resid_lim, resid_lim)
        ax_resid.set_title(f"{cat_name} -- residual (observed - median)")
        ax_resid.grid(True, alpha=0.3)
 
        # --- normal-scale curve panel (right) ---
        for i in idxs:
            has = ~np.isnan(observed[i])
            ax_curve.plot(t[has], observed[i][has], color=color, alpha=0.3, linewidth=0.7)
            ax_curve.plot(t, median[i], color=color, linewidth=1.8)
            ax_curve.plot(t, upper[i], color=color, linewidth=1.2, linestyle="--")
        if fit.cutoff != "max":
            ax_curve.axvspan(0.0, t[fit.cutoff], color="k", alpha=0.06)
        ax_curve.set_xlim(left=0.0)
        ax_curve.set_ylim(0, 1)
        ax_curve.set_title(cat_name)
        ax_curve.grid(True, alpha=0.3)
 
    for ax in axes[:, 0]:
        ax.set_ylabel("Residual")
    for ax in axes[:, 1]:
        ax.set_ylabel(y_label)
    for ax in axes[-1, :]:
        ax.set_xlabel("Step")
 
    fig.suptitle(title or f"Hierarchical success-probability forecast + residuals (cutoff={fit.cutoff})")
    fig.tight_layout()
    plt.close(fig)
    return fig