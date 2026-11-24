"""
Curve-fitting/forecasting for attack trajectories, notebook-callable.

Mandatory entry point: `fit_and_forecast(data, families=None, cutoffs=("max",))`.
`data` is a 2D `(n_behaviours, n_steps)` array (one score/ASR/hazard trajectory
per row; loading and aggregating raw scores into this shape is the caller's
job). Columns are step indices; unobserved steps are NaN and are dropped
per-row while preserving true spacing on the time axis.

All signals are assumed bounded in [0, 1] and generally (not strictly)
increasing. Every family's asymptote is bounded to [0, 1] accordingly, and
continuous families additionally warn (but do not reject) fits that dip
below 0 at t=0 -- see `soft_validate_fn` below.

Two family groups:
  * CONTINUOUS_FAMILY_NAMES -- no assumption about f(0); use for mean judge
    score and hazard, which have a nonzero baseline.
  * RATE_FAMILY_NAMES -- hard-code f(0)=0; only meaningful for ASR.
Fit both to any row if you want held-out error to arbitrate, but a rate family
fit to a row with a nonzero baseline will lose for structural reasons.

For each requested cutoff, every requested family is fit to the row's observed
values at `t <= cutoff` (or, for `cutoff="max"`, to every observed value) with
a small random-restart multistart, and forecast quality is evaluated on
whatever is held out beyond the cutoff (nothing, for `"max"`). Returns a single
long-format DataFrame.

Note on cutoffs: they are *column indices*, not counts of observations. With
intermittent sampling use `observed_columns(data)` to pick them, e.g.
`obs = observed_columns(data); cutoffs = [obs[len(obs) // 2]]`.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import expit

__all__ = [
    "ALL_FAMILY_NAMES",
    "CONTINUOUS_FAMILY_NAMES",
    "FAMILY_REGISTRY",
    "RATE_FAMILY_NAMES",
    "CurveFamily",
    "fit_and_forecast",
    "observed_columns",
]

logger = logging.getLogger(__name__)


def _ensure_logging_configured(log_level: str) -> None:
    # loky worker processes start fresh (no inherited handlers), so without
    # this, warnings emitted inside a parallel task would silently vanish.
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(log_level)


def observed_columns(data: np.ndarray) -> np.ndarray:
    """Column indices of `data` holding at least one observed (non-NaN) value."""
    return np.flatnonzero(~np.isnan(np.asarray(data, dtype=float)).all(axis=0))


# =============================================================================
# Candidate curve families.
#
# Asymptote parameters are bounded to [0, 1] because all three signal types
# are. This is what stops a sigmoid from placing its lower plateau at negative
# y: with only a couple of near-zero observations before the rise, the SSE
# surface is nearly flat in the lower asymptote, so an unbounded L drifts
# freely and the fit renders as "the top half of an S".
#
# Bounds that are genuine physical boundaries (0 and 1) are listed in
# `pin_exempt`: a fit landing exactly there is legitimate, not a sign the
# optimizer ran out of room. Regularization bounds (positivity on rate
# constants, kappa in [0.1, 5]) are *not* exempt -- hitting those still
# invalidates the fit.
# =============================================================================

# Approximates a strict ">0" bound so curve_fit's inclusive box constraints
# still forbid the parameter from landing exactly on zero.
_POS_EPS = 1e-8
# Slack allowed on the f(0) >= 0 check, in signal units.
_START_TOL = 0.02


def _t_range(t_fit: np.ndarray) -> float:
    span = float(np.max(t_fit) - np.min(t_fit))
    return span if span > 1e-8 else 1e-8


def _rates(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference rates and their interval midpoints, spacing-aware."""
    dt = np.diff(t)
    return np.diff(y) / dt, t[:-1] + dt / 2.0


@dataclass(frozen=True)
class CurveFamily:
    name: str
    func: Callable[..., np.ndarray]
    param_names: tuple[str, ...]
    lower_bounds: Callable[[np.ndarray, np.ndarray], tuple[float, ...]]
    upper_bounds: Callable[[np.ndarray, np.ndarray], tuple[float, ...]]
    init_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]]
    is_rate_family: bool
    asymptote_param_name: str | None
    doc: str = field(default="", repr=False)
    # Extra validity check beyond box bounds (e.g. logistic's U>L), which per-
    # parameter bounds can't express. A non-None return REJECTS the attempt
    # (the next-best multistart attempt is tried instead).
    validate_fn: Callable[[dict[str, float]], str | None] = field(
        default=lambda params: None, repr=False
    )
    # Same signature as validate_fn, but a non-None return does NOT reject the
    # attempt -- it's kept as a converged fit, just tagged with the returned
    # string in the `validation_warning` column. For checks that flag a
    # plausible-but-imperfect fit (e.g. a slightly-off start) rather than a
    # structurally invalid one.
    soft_validate_fn: Callable[[dict[str, float]], str | None] | None = field(
        default=None, repr=False
    )
    # Params allowed to sit on a bound without failing the fit.
    pin_exempt: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def bounds(self, t_fit, y_fit) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self.lower_bounds(t_fit, y_fit), self.upper_bounds(t_fit, y_fit)

    def eval(self, t: np.ndarray, params: dict[str, float]) -> np.ndarray:
        return self.func(t, *(params[name] for name in self.param_names))


def _reject_negative_start(params: dict[str, float]) -> str | None:
    """For the a - b*g(t) families: f(0) = a - b must not be meaningfully < 0.
    Registered as a soft check (see CurveFamily.soft_validate_fn) -- a curve
    that undershoots at t=0 to better match a delayed-onset rise later on is
    still a usable fit if you only care about the asymptote."""
    return "negative_at_t0" if params["a"] - params["b"] < -_START_TOL else None


def _exponential(t, a, b, c):
    return a - b * np.exp(-c * t)


def _init_exponential(t_fit, y_fit):
    a0 = min(float(np.nanmax(y_fit)) * 1.05, 1.0)
    return {"a": a0, "b": max(a0 - float(y_fit[0]), 1e-3), "c": 3.0 / _t_range(t_fit)}


FAMILY_exponential = CurveFamily(
    name="exponential",
    func=_exponential,
    param_names=("a", "b", "c"),
    lower_bounds=lambda t, y: (0.0, _POS_EPS, _POS_EPS),
    upper_bounds=lambda t, y: (1.0, np.inf, np.inf),
    init_fn=_init_exponential,
    is_rate_family=False,
    asymptote_param_name="a",
    doc="f(t) = a - b*exp(-c*t)",
    soft_validate_fn=_reject_negative_start,
    pin_exempt=frozenset({"a"}),
)


def _stretched_exponential(t, a, b, c, kappa):
    return a - b * np.exp(-((c * t) ** kappa))


def _init_stretched_exponential(t_fit, y_fit):
    return {**_init_exponential(t_fit, y_fit), "kappa": 1.0}


FAMILY_stretched_exponential = CurveFamily(
    name="stretched_exponential",
    func=_stretched_exponential,
    param_names=("a", "b", "c", "kappa"),
    lower_bounds=lambda t, y: (0.0, _POS_EPS, _POS_EPS, 0.1),
    upper_bounds=lambda t, y: (1.0, np.inf, np.inf, 5.0),
    init_fn=_init_stretched_exponential,
    is_rate_family=False,
    asymptote_param_name="a",
    doc="f(t) = a - b*exp(-(c*t)^kappa), nests `exponential` at kappa=1",
    soft_validate_fn=_reject_negative_start,
    pin_exempt=frozenset({"a"}),
)


def _power_law_saturating(t, a, b, c, nu):
    return a - b * (1.0 + c * t) ** (-nu)


def _init_power_law_saturating(t_fit, y_fit):
    a0 = min(float(np.nanmax(y_fit)) * 1.05, 1.0)
    return {
        "a": a0,
        "b": max(a0 - float(y_fit[0]), 1e-3),
        "c": 3.0 / _t_range(t_fit),
        "nu": 1.0,
    }


FAMILY_power_law_saturating = CurveFamily(
    name="power_law_saturating",
    func=_power_law_saturating,
    param_names=("a", "b", "c", "nu"),
    lower_bounds=lambda t, y: (0.0, _POS_EPS, _POS_EPS, _POS_EPS),
    upper_bounds=lambda t, y: (1.0, np.inf, np.inf, np.inf),
    init_fn=_init_power_law_saturating,
    is_rate_family=False,
    asymptote_param_name="a",
    doc="f(t) = a - b*(1+c*t)^(-nu)",
    soft_validate_fn=_reject_negative_start,
    pin_exempt=frozenset({"a"}),
)


def _logistic(t, L, U, kappa, t0):
    return L + (U - L) * expit(kappa * (t - t0))


def _init_logistic(t_fit, y_fit):
    lo, hi = float(np.nanmin(y_fit)), float(np.nanmax(y_fit))
    # Start t0 at the steepest observed rise rather than the median step: for
    # these trajectories the transition is often early, and a median start
    # plus multiplicative jitter explores the wrong region entirely.
    if len(y_fit) >= 2:
        rate, mid = _rates(t_fit, y_fit)
        t0 = float(mid[int(np.argmax(rate))])
    else:
        t0 = float(np.median(t_fit))
    return {
        "L": lo,
        "U": min(max(hi * 1.05, lo + 1e-3), 1.0),
        "kappa": 4.0 / _t_range(t_fit),
        "t0": t0,
    }


def _validate_logistic(params: dict[str, float]) -> str | None:
    return "invalid_param_ordering" if params["U"] <= params["L"] else None


FAMILY_logistic = CurveFamily(
    name="logistic",
    func=_logistic,
    param_names=("L", "U", "kappa", "t0"),
    # L in [0, 1] is what keeps the lower plateau on-scale; U's lower bound is
    # pinned at the observed min as a data-driven nudge toward U>L, with the
    # actual joint constraint enforced post-fit via validate_fn. With L >= 0
    # and U > L, f(0) >= 0 comes for free.
    lower_bounds=lambda t, y: (0.0, float(np.nanmin(y)), _POS_EPS, np.min(t) - _t_range(t)),
    upper_bounds=lambda t, y: (1.0, 1.0, np.inf, np.max(t) + _t_range(t)),
    init_fn=_init_logistic,
    is_rate_family=False,
    asymptote_param_name="U",
    doc="f(t) = L + (U-L) / (1 + exp(-kappa*(t-t0))); L,U in [0,1], U>L enforced post-fit",
    # Kept as a hard check: U<=L is a structurally invalid parameterization
    # (not a "shape mismatch at the start" the way negative_at_t0 is), so
    # there's no sense in which the resulting fit is usable.
    validate_fn=_validate_logistic,
    pin_exempt=frozenset({"L", "U"}),
)


# def _log_linear(t, a, b):
#     return a + b * np.log1p(t)


# def _init_log_linear(t_fit, y_fit):
#     denom = np.log1p(float(t_fit[-1])) or 1e-12
#     return {"a": float(y_fit[0]), "b": (float(y_fit[-1]) - float(y_fit[0])) / denom}


# FAMILY_log_linear = CurveFamily(
#     name="log_linear",
#     func=_log_linear,
#     param_names=("a", "b"),
#     # `a` = f(0) is bounded to the signal range; `b` is deliberately left free
#     # so the curve can run past 1 -- this family is the no-ceiling control.
#     lower_bounds=lambda t, y: (0.0, -np.inf),
#     upper_bounds=lambda t, y: (1.0, np.inf),
#     init_fn=_init_log_linear,
#     is_rate_family=False,
#     asymptote_param_name=None,  # no ceiling -- a no-asymptote falsification control
#     doc="f(t) = a + b*log(1+t); has no asymptote",
#     pin_exempt=frozenset({"a"}),
# )


# def _cure_exponential(t, pi, c):
#     return pi * (1.0 - np.exp(-c * t))


# def _init_cure_exponential(t_fit, y_fit):
#     pi0 = max(min(1.0, float(np.nanmax(y_fit)) * 1.1), 1e-3)
#     return {"pi": pi0, "c": 3.0 / _t_range(t_fit)}


# FAMILY_cure_exponential = CurveFamily(
#     name="cure_exponential",
#     func=_cure_exponential,
#     param_names=("pi", "c"),
#     lower_bounds=lambda t, y: (_POS_EPS, _POS_EPS),
#     upper_bounds=lambda t, y: (1.0, np.inf),
#     init_fn=_init_cure_exponential,
#     is_rate_family=True,
#     asymptote_param_name="pi",
#     doc="ASR(t) = pi*(1-exp(-c*t)); assumes f(0)=0",
#     # pi=1 is full saturation, a legitimate outcome, not a stuck optimizer.
#     pin_exempt=frozenset({"pi"}),
# )


# def _cure_weibull(t, pi, c, kappa):
#     return pi * (1.0 - np.exp(-((c * t) ** kappa)))


# def _init_cure_weibull(t_fit, y_fit):
#     return {**_init_cure_exponential(t_fit, y_fit), "kappa": 1.0}


# FAMILY_cure_weibull = CurveFamily(
#     name="cure_weibull",
#     func=_cure_weibull,
#     param_names=("pi", "c", "kappa"),
#     lower_bounds=lambda t, y: (_POS_EPS, _POS_EPS, 0.1),
#     upper_bounds=lambda t, y: (1.0, np.inf, 5.0),
#     init_fn=_init_cure_weibull,
#     is_rate_family=True,
#     asymptote_param_name="pi",
#     doc="ASR(t) = pi*(1-exp(-(c*t)^kappa)), nests `cure_exponential` at kappa=1",
#     pin_exempt=frozenset({"pi"}),
# )


# def _power_exponential_crossover_rate(t, pi, c, nu, mu):
#     return pi * (1.0 - (1.0 + c * t) ** (-nu) * np.exp(-mu * t))


# def _init_power_exponential_crossover_rate(t_fit, y_fit):
#     pi0 = max(min(1.0, float(np.nanmax(y_fit)) * 1.1), 1e-3)
#     return {"pi": pi0, "c": 3.0 / _t_range(t_fit), "nu": 1.0, "mu": 0.0}


# FAMILY_power_exponential_crossover_rate = CurveFamily(
#     name="power_exponential_crossover",
#     func=_power_exponential_crossover_rate,
#     param_names=("pi", "c", "nu", "mu"),
#     lower_bounds=lambda t, y: (_POS_EPS, _POS_EPS, 0.0, 0.0),
#     upper_bounds=lambda t, y: (1.0, np.inf, np.inf, np.inf),
#     init_fn=_init_power_exponential_crossover_rate,
#     is_rate_family=True,
#     asymptote_param_name="pi",
#     doc="ASR(t) = pi*(1 - (1+c*t)^(-nu)*exp(-mu*t)); assumes f(0)=0, pi is the free asymptote",
#     # pi=1 is full saturation, a legitimate outcome, not a stuck optimizer
#     # (mirrors cure_exponential/cure_weibull). mu=0 nests the pure power-law
#     # case; nu=0 stays subject to the pin check, same reasoning as before.
#     pin_exempt=frozenset({"pi", "mu", "c", "nu"}),
# )


def _power_exponential_crossover(t, a, b, c, nu, mu):
    return a - b * (1.0 + c * t) ** (-nu) * np.exp(-mu * t)

def _init_power_exponential_crossover(t_fit, y_fit):
    a0 = min(float(np.nanmax(y_fit)) * 1.05, 1.0)
    return {
        "a": a0,
        "b": max(a0 - float(y_fit[0]), 1e-3),
        "c": 3.0 / _t_range(t_fit),
        "nu": 1.0,
        "mu": 0.0,
    }

FAMILY_power_exponential_crossover = CurveFamily(
    name="power_exponential_crossover",
    func=_power_exponential_crossover,
    param_names=("a", "b", "c", "nu", "mu"),
    lower_bounds=lambda t, y: (0.0, _POS_EPS, _POS_EPS, 0.0, 0.0),
    upper_bounds=lambda t, y: (1.0, np.inf, np.inf, np.inf, np.inf),
    init_fn=_init_power_exponential_crossover,
    is_rate_family=False,
    asymptote_param_name="a",
    doc="f(t) = a - b*(1+c*t)^(-nu)*exp(-mu*t); f(0) = a-b is free",
    pin_exempt=frozenset({"a", "b", "mu", "c", "nu"}),
)

FAMILY_REGISTRY: dict[str, CurveFamily] = {
    f.name: f
    for f in [
        FAMILY_exponential,
        FAMILY_stretched_exponential,
        FAMILY_power_law_saturating,
        FAMILY_logistic,
        # FAMILY_log_linear,
        # FAMILY_cure_exponential,
        # FAMILY_cure_weibull,
        # FAMILY_power_exponential_crossover_rate,
        FAMILY_power_exponential_crossover
    ]
}
ALL_FAMILY_NAMES: tuple[str, ...] = tuple(FAMILY_REGISTRY.keys())
CONTINUOUS_FAMILY_NAMES: tuple[str, ...] = tuple(
    n for n, f in FAMILY_REGISTRY.items() if not f.is_rate_family
)
RATE_FAMILY_NAMES: tuple[str, ...] = tuple(n for n, f in FAMILY_REGISTRY.items() if f.is_rate_family)


# =============================================================================
# Curvature diagnostic: does the fit window show a statistically significant
# decelerating trend? Computed once per (row, cutoff), independent of family.
# Rates are divided by the actual step spacing, so uneven observation grids
# don't get read as changes in slope.
# =============================================================================

_MIN_DIFF_POINTS = 4


def _curvature_detected(t_fit: np.ndarray, y_fit: np.ndarray) -> bool:
    if len(y_fit) < _MIN_DIFF_POINTS + 1:
        return False
    rate, mid = _rates(t_fit, y_fit)
    result = stats.linregress(mid, rate)
    if not np.isfinite(result.slope) or not np.isfinite(result.pvalue) or result.slope >= 0:
        return False
    # p-value is two-sided; halve for the one-sided (slope < 0) test, since
    # slope < 0 is already gated above.
    return bool(result.pvalue / 2.0 < 0.05)


# =============================================================================
# Fixed-restart multistart fit
# =============================================================================

_N_MULTISTART_ATTEMPTS = 100
_BOUND_PIN_RTOL = 1e-6


def _stable_int_hash(s: str) -> int:
    # Deterministic (PYTHONHASHSEED-independent) string hash, for reproducible RNG seeding.
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:16], 16)


def _derive_rng(random_seed: int, row_id: str, family_name: str) -> np.random.Generator:
    # Deliberately excludes cutoff: every cutoff for a given (row, family)
    # explores the same sequence of multistart perturbations, so whether a
    # fit converges at one cutoff but not another reflects the data changing,
    # not an unrelated random draw.
    return np.random.default_rng(
        (int(random_seed), _stable_int_hash(str(row_id)), _stable_int_hash(family_name))
    )


@dataclass
class _FitOutcome:
    converged: bool
    failure_reason: str | None
    params: dict[str, float] | None
    param_stderr: dict[str, float] | None
    n_multistart_attempts: int
    in_sample_sse: float | None
    validation_warning: str | None
    failure_detail: str | None = None


def _perturb_init(init: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    out = {}
    for name, value in init.items():
        # multiplicative jitter can't move a param away from exactly zero.
        out[name] = (
            float(rng.normal(loc=0.0, scale=0.05))
            if abs(value) < 1e-8
            else float(value * rng.lognormal(mean=0.0, sigma=0.3))
        )
    return out


def _clip_into_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.where(np.isfinite(upper - lower), upper - lower, 1.0)
    eps = 1e-6 * np.maximum(np.abs(span), 1.0)
    lo_eff = np.where(np.isfinite(lower), lower + eps, -np.inf)
    hi_eff = np.where(np.isfinite(upper), upper - eps, np.inf)
    return np.clip(x, lo_eff, hi_eff)


def _fit_once(
    family: CurveFamily, t_fit: np.ndarray, y_fit: np.ndarray, init: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, float] | None:
    lower, upper = family.bounds(t_fit, y_fit)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    p0 = _clip_into_bounds(
        np.array([init[name] for name in family.param_names], dtype=float), lower, upper
    )
    try:
        with warnings.catch_warnings():
            # ill-conditioned perturbed starts are an expected, benign part
            # of multistart exploration; a genuinely failed fit still raises.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=OptimizeWarning)
            popt, pcov = curve_fit(
                family.func,
                t_fit,
                y_fit,
                p0=p0,
                bounds=(lower, upper),
                maxfev=10_000,
                # Parameters differ by orders of magnitude (logistic's L,U are
                # on the signal scale; kappa ~ 1/t-scale; t0 ~ t-scale), which
                # `trf` handles badly at the default unit scaling.
                loss='soft_l1',
                method='trf',
                x_scale="jac",
            )
    except (RuntimeError, ValueError):
        return None
    pred = family.func(t_fit, *popt)
    if not (np.all(np.isfinite(popt)) and np.all(np.isfinite(pred))):
        return popt, pcov, float("inf")
    return popt, pcov, float(np.sum((y_fit - pred) ** 2))


def _check_attempt(
    family: CurveFamily, popt: np.ndarray, t_fit: np.ndarray, y_fit: np.ndarray
) -> tuple[dict[str, float] | None, str | None, str | None, tuple[str, ...]]:
    """Returns (params, hard_reason, soft_warning, pinned_params).

    hard_reason set (params is None) -> attempt rejected outright; the next-
    best multistart attempt (by SSE) is tried instead.
    soft_warning set (params is not None) -> attempt accepted as a converged
    fit, but flagged in the `validation_warning` output column.
    pinned_params lists every non-exempt parameter that landed on its bound
    (with which bound and its value), when hard_reason == "param_at_bound";
    empty otherwise. Diagnostic detail for callers, e.g. to tell "the curve
    wants to be flat" (c or nu pinned near 0) apart from other degeneracies.
    """
    if not np.all(np.isfinite(popt)):
        return None, "non_finite_params", None, ()
    params = dict(zip(family.param_names, popt, strict=True))
    lower, upper = family.bounds(t_fit, y_fit)
    pinned = []
    for name, value, lo, hi in zip(family.param_names, popt, lower, upper, strict=True):
        if name in family.pin_exempt:
            continue
        tol_lo = _BOUND_PIN_RTOL * max(abs(lo), 1.0)
        tol_hi = _BOUND_PIN_RTOL * max(abs(hi), 1.0)
        if np.isfinite(lo) and abs(value - lo) < tol_lo:
            pinned.append(f"{name}@lo={lo:.3g}")
        elif np.isfinite(hi) and abs(value - hi) < tol_hi:
            pinned.append(f"{name}@hi={hi:.3g}")
    if pinned:
        return None, "param_at_bound", None, tuple(pinned)
    hard_reason = family.validate_fn(params)
    if hard_reason is not None:
        return None, hard_reason, None, ()
    soft_warning = family.soft_validate_fn(params) if family.soft_validate_fn else None
    return params, None, soft_warning, ()


def _multistart_fit(
    family: CurveFamily, t_fit: np.ndarray, y_fit: np.ndarray, rng: np.random.Generator
) -> _FitOutcome:
    """
    Fixed number of restarts; returns the lowest-SSE attempt that also passes
    the bound/hard-validity checks. Checking only the single best attempt (and
    failing the whole cell if it happens to sit on a bound) discards perfectly
    good runner-up fits, so all converged attempts are considered in SSE order.
    The unperturbed data-derived init is always among them.
    """
    base_init = family.init_fn(t_fit, y_fit)
    inits = [base_init] + [
        _perturb_init(base_init, rng) for _ in range(_N_MULTISTART_ATTEMPTS - 1)
    ]
    attempts = [_fit_once(family, t_fit, y_fit, init) for init in inits]
    n_attempts = len(attempts)
    converged = sorted((a for a in attempts if a is not None), key=lambda a: a[2])

    if not converged:
        detail = f"0/{n_attempts} restarts produced a finite fit (curve_fit raised or diverged every time)"
        return _FitOutcome(False, "optimizer_did_not_converge", None, None, n_attempts, None, None, detail)

    first_reason = None
    reason_counts: Counter[str] = Counter()
    pin_counts: Counter[str] = Counter()
    for popt, pcov, sse in converged:
        params, hard_reason, soft_warning, pinned = _check_attempt(family, popt, t_fit, y_fit)
        if hard_reason is not None:
            reason_counts[hard_reason] += 1
            pin_counts.update(pinned)
            first_reason = first_reason or hard_reason
            continue
        if pcov is not None and np.all(np.isfinite(pcov)):
            stderr = dict(
                zip(family.param_names, np.sqrt(np.clip(np.diag(pcov), 0, None)), strict=True)
            )
        else:
            stderr = dict.fromkeys(family.param_names, float("nan"))
        return _FitOutcome(True, None, params, stderr, n_attempts, sse, soft_warning, None)

    detail = (
        f"{len(converged)}/{n_attempts} restarts converged numerically but every one was rejected "
        f"({', '.join(f'{r}x{c}' for r, c in reason_counts.most_common())})"
    )
    if pin_counts:
        detail += "; bound hit by: " + ", ".join(f"{p}x{c}" for p, c in pin_counts.most_common())
    return _FitOutcome(False, first_reason, None, None, n_attempts, None, None, detail)


# =============================================================================
# Per-(row, family, cutoff) task and orchestration
# =============================================================================

_RESULT_COLUMNS = (
    "row_id",
    "family",
    "cutoff",
    "converged",
    "failure_reason",
    "failure_detail",
    "validation_warning",
    "params",
    "param_stderr",
    "n_fit_points",
    "n_holdout_points",
    "n_multistart_attempts",
    "fit_wall_time_seconds",
    "in_sample_rmse",
    "holdout_mae",
    "holdout_rmse",
    "holdout_mae_normalized",
    "holdout_rmse_normalized",
    "holdout_endpoint_error",
    "aic",
    "asymptote_param_name",
    "pseudo_ground_truth_asymptote",
    "asymptote_abs_error",
    "asymptote_rel_error",
    "curvature_detected",
    "predicted_curve",
)


def _fit_one(
    row_id: str,
    family_name: str,
    cutoff: int | str,
    t_values: np.ndarray,
    y_row: np.ndarray,
    min_fit_points: int,
    random_seed: int,
    log_level: str,
) -> dict:
    """
    `cutoff="max"` fits every observed point of the row (no holdout split);
    holdout_* and n_holdout_points are left None for these rows. A numeric
    cutoff fits `t <= cutoff` and evaluates holdout metrics on `t > cutoff`;
    its asymptote ground-truth fields are filled in by `fit_and_forecast`
    after all tasks complete.
    """
    _ensure_logging_configured(log_level)
    start = time.perf_counter()
    family = FAMILY_REGISTRY[family_name]
    n_T = len(t_values)
    is_full = cutoff == "max"

    if is_full:
        valid_fit = ~np.isnan(y_row)
        t_fit, y_fit = t_values[valid_fit], y_row[valid_fit]
        t_hold, y_hold, n_holdout_points = np.array([]), np.array([]), None
    else:
        fit_mask = np.arange(n_T) <= cutoff
        t_fit_all, y_fit_all = t_values[fit_mask], y_row[fit_mask]
        valid_fit = ~np.isnan(y_fit_all)
        t_fit, y_fit = t_fit_all[valid_fit], y_fit_all[valid_fit]

        t_hold_all, y_hold_all = t_values[~fit_mask], y_row[~fit_mask]
        valid_hold = ~np.isnan(y_hold_all)
        t_hold, y_hold = t_hold_all[valid_hold], y_hold_all[valid_hold]
        n_holdout_points = int(valid_hold.sum())

    n_fit_points = int(len(y_fit))
    curv = _curvature_detected(t_fit, y_fit)

    def _failed(reason: str, n_attempts: int, detail: str | None = None) -> dict:
        logger.warning(
            "fit failed: row_id=%r family=%r cutoff=%r reason=%s (n_fit_points=%d)%s",
            row_id, family_name, cutoff, reason, n_fit_points,
            f" -- {detail}" if detail else "",
        )
        return {
            "row_id": row_id,
            "family": family_name,
            "cutoff": cutoff,
            "converged": False,
            "failure_reason": reason,
            "failure_detail": detail,
            "validation_warning": None,
            "params": None,
            "param_stderr": None,
            "n_fit_points": n_fit_points,
            "n_holdout_points": n_holdout_points,
            "n_multistart_attempts": n_attempts,
            "fit_wall_time_seconds": time.perf_counter() - start,
            "in_sample_rmse": None,
            "holdout_mae": None,
            "holdout_rmse": None,
            "holdout_mae_normalized": None,
            "holdout_rmse_normalized": None,
            "holdout_endpoint_error": None,
            "aic": None,
            "asymptote_param_name": family.asymptote_param_name,
            "pseudo_ground_truth_asymptote": None,
            "asymptote_abs_error": None,
            "asymptote_rel_error": None,
            "curvature_detected": curv,
            "predicted_curve": np.full(n_T, np.nan),
        }

    required_min = max(min_fit_points, family.n_params + 1)
    if n_fit_points < required_min:
        return _failed(
            "insufficient_data", 0, detail=f"need >= {required_min} fit points, got {n_fit_points}"
        )

    rng = _derive_rng(random_seed, row_id, family_name)
    outcome = _multistart_fit(family, t_fit, y_fit, rng)
    if not outcome.converged:
        return _failed(
            outcome.failure_reason, outcome.n_multistart_attempts, detail=outcome.failure_detail
        )

    if outcome.validation_warning:
        logger.info(
            "fit converged with a soft validation warning: row_id=%r family=%r cutoff=%r "
            "warning=%s (n_fit_points=%d)",
            row_id, family_name, cutoff, outcome.validation_warning, n_fit_points,
        )

    predicted_curve = family.eval(t_values, outcome.params)
    in_sample_pred = family.eval(t_fit, outcome.params)
    in_sample_rmse = float(np.sqrt(np.mean((y_fit - in_sample_pred) ** 2)))

    if not is_full and n_holdout_points > 0:
        holdout_pred = family.eval(t_hold, outcome.params)
        errs = y_hold - holdout_pred
        holdout_mae = float(np.mean(np.abs(errs)))
        holdout_rmse = float(np.sqrt(np.mean(errs**2)))

        row_valid = y_row[~np.isnan(y_row)]
        row_range = float(np.max(row_valid) - np.min(row_valid)) if len(row_valid) else 0.0
        # normalize by the row's observed range; fall back to |mean| (never
        # raw unnormalized MAE) for a near-constant row.
        norm = row_range if row_range > 1e-8 else max(abs(float(np.mean(row_valid))), 1e-8)
        holdout_mae_normalized = holdout_mae / norm
        holdout_rmse_normalized = holdout_rmse / norm

        n_endpoint = min(5, n_holdout_points)
        endpoint_pred = family.eval(t_hold[-n_endpoint:], outcome.params)
        holdout_endpoint_error = float(np.mean(np.abs(y_hold[-n_endpoint:] - endpoint_pred)))
    else:
        holdout_mae = holdout_rmse = holdout_mae_normalized = holdout_rmse_normalized = None
        holdout_endpoint_error = None

    # Simplified Gaussian AIC -- adequate for cross-family relative
    # comparison within a (row_id, cutoff) group, not exact-likelihood AIC.
    aic = n_fit_points * np.log(max(outcome.in_sample_sse / n_fit_points, 1e-300)) + 2 * family.n_params

    return {
        "row_id": row_id,
        "family": family_name,
        "cutoff": cutoff,
        "converged": True,
        "failure_reason": None,
        "failure_detail": None,
        "validation_warning": outcome.validation_warning,
        "params": outcome.params,
        "param_stderr": outcome.param_stderr,
        "n_fit_points": n_fit_points,
        "n_holdout_points": n_holdout_points,
        "n_multistart_attempts": outcome.n_multistart_attempts,
        "fit_wall_time_seconds": time.perf_counter() - start,
        "in_sample_rmse": in_sample_rmse,
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "holdout_mae_normalized": holdout_mae_normalized,
        "holdout_rmse_normalized": holdout_rmse_normalized,
        "holdout_endpoint_error": holdout_endpoint_error,
        "aic": float(aic),
        "asymptote_param_name": family.asymptote_param_name,
        "pseudo_ground_truth_asymptote": None,
        "asymptote_abs_error": None,
        "asymptote_rel_error": None,
        "curvature_detected": curv,
        "predicted_curve": predicted_curve,
    }


def _resolve_cutoffs(
    cutoffs: Sequence[int | str], obs_cols: np.ndarray, min_fit_points: int
) -> list[int | str]:
    """
    Validate cutoffs against the *observed* grid, not the column count. With
    intermittent sampling a cutoff can look valid by column index while
    leaving only one or two actual observations in the fit window.
    """
    if not cutoffs:
        raise ValueError("`cutoffs` must be non-empty.")
    resolved: list[int | str] = []
    seen: set[int | str] = set()
    for c in cutoffs:
        if isinstance(c, str):
            if c != "max":
                raise ValueError(f"String cutoff values must be 'max'; got {c!r}")
            key: int | str = "max"
        else:
            key = int(c)
        if key not in seen:
            seen.add(key)
            resolved.append(key)

    if len(obs_cols) < min_fit_points + 1:
        raise ValueError(
            f"data has only {len(obs_cols)} observed columns; need at least "
            f"{min_fit_points + 1} for any numeric cutoff."
        )
    valid_lo, valid_hi = int(obs_cols[min_fit_points - 1]), int(obs_cols[-2])
    bad = []
    for c in resolved:
        if not isinstance(c, int):
            continue
        n_in = int(np.sum(obs_cols <= c))
        n_out = int(np.sum(obs_cols > c))
        if n_in < min_fit_points or n_out < 1:
            bad.append(f"{c} ({n_in} observed points in window, {n_out} held out)")
    if bad:
        raise ValueError(
            "numeric cutoffs are column indices and must leave >= "
            f"{min_fit_points} observed points in the fit window and >= 1 held out. "
            f"Valid range for this data: [{valid_lo}, {valid_hi}]. Invalid: {bad}. "
            "Use `observed_columns(data)` to pick cutoffs on the observation grid."
        )
    return resolved


def _warn_about_data(data: np.ndarray, families: Sequence[str], include_aggregate: bool) -> None:
    finite = data[~np.isnan(data)]
    if len(finite) and (finite.min() < -1e-9 or finite.max() > 1.0 + 1e-9):
        logger.warning(
            "data range [%.3f, %.3f] falls outside [0, 1]; asymptote parameters are bounded "
            "to [0, 1] and fits will be truncated. Rescale the data or relax the bounds.",
            float(finite.min()), float(finite.max()),
        )

    if any(FAMILY_REGISTRY[f].is_rate_family for f in families):
        first_obs = []
        for row in data:
            idx = np.flatnonzero(~np.isnan(row))
            if len(idx):
                first_obs.append(float(row[idx[0]]))
        if first_obs and float(np.mean(first_obs)) > 0.05:
            logger.warning(
                "rate families (%s) hard-code f(0)=0 but the mean first observed value is %.3f; "
                "they will fit a nonzero baseline poorly for structural rather than shape reasons.",
                ", ".join(sorted(set(families) & set(RATE_FAMILY_NAMES))), float(np.mean(first_obs)),
            )

    if include_aggregate:
        counts = (~np.isnan(data)).sum(axis=0)
        observed = counts[counts > 0]
        if len(observed) and len(np.unique(observed)) > 1:
            logger.warning(
                "rows are observed on different step grids (per-column counts range %d-%d); the "
                "`aggregate` row averages a different subset of behaviours per column and is an "
                "artifact. Consider include_aggregate=False.",
                int(observed.min()), int(observed.max()),
            )


def fit_and_forecast(
    data: np.ndarray,
    families: Sequence[str] | None = None,
    cutoffs: Sequence[int | str] = ("max",),
    t_values: np.ndarray | None = None,
    row_ids: Sequence[str] | None = None,
    include_aggregate: bool = True,
    min_fit_points: int = 4,
    n_jobs: int = -1,
    random_seed: int = 0,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Fit every family in `families` to every row of `data` at every cutoff in
    `cutoffs`, in parallel.

    Args:
        data: `(n_behaviours, n_steps)` array of trajectories in [0, 1].
            Unobserved steps are NaN; they are dropped per row while keeping
            the true spacing on the time axis.
        families: curve family names to fit. Defaults to all 8. Use
            `CONTINUOUS_FAMILY_NAMES` for judge score / hazard (nonzero
            baseline) and add `RATE_FAMILY_NAMES` for ASR (f(0)=0).
        cutoffs: a mix of non-negative ints (fit `t <= cutoff`, evaluate
            holdout on `t > cutoff`) and/or the string `"max"` (fit every
            observed point of the row, no holdout). Defaults to `["max"]`.
            Cutoffs are *column indices*: pick them from `observed_columns(data)`.
        t_values: step index for each column of `data`. Defaults to
            `arange(n_steps)`. Pass the real step numbers if columns are not
            evenly spaced.
        row_ids: label for each row of `data`. Defaults to `"0", "1", ...`.
        include_aggregate: also fit an extra `row_id="aggregate"` row equal to
            `nanmean(data, axis=0)`. Only meaningful if all rows share an
            observation grid (a warning is logged if they don't).
        min_fit_points: minimum observed points required in the fit window (the
            actual minimum used is `max(min_fit_points, family.n_params + 1)`).
        n_jobs: passed to `joblib.Parallel` (-1 = all cores).
        random_seed: seeds the deterministic per-(row, family) multistart RNG
            (shared across all cutoffs for that row/family -- see `_derive_rng`).
        log_level: logging level for this module during the run.

    Returns:
        Long-format DataFrame, one row per (row_id, family, cutoff), with the
        columns listed in `_RESULT_COLUMNS`.

        For `cutoff="max"` rows (full-data fits): n_holdout_points and every
        holdout_*/asymptote-ground-truth field is None -- neither concept
        applies to a fit on the whole curve.

        For numeric-cutoff rows: `pseudo_ground_truth_asymptote` /
        `asymptote_abs_error` / `asymptote_rel_error` are filled in by
        looking up the `cutoff="max"` fit for the same (row_id, family) --
        only if `"max"` is itself among the requested `cutoffs`, that fit
        converged, and the family has an asymptote parameter; otherwise None.

        `converged=True` rows may still carry a non-None `validation_warning`
        (currently only `"negative_at_t0"`, for exponential/stretched_
        exponential/power_law_saturating): the fit was kept because the soft
        check doesn't invalidate it, but the curve's behaviour at t=0 doesn't
        match the data well. Fine to ignore if you only care about the
        asymptote; worth a look if you care about the early-time shape.

        `df.attrs` carries `random_seed`, `families`, `cutoffs`,
        `total_wall_time_seconds`, `sum_fit_wall_time_seconds`.

        Rank families within a (row_id, cutoff) group. Compare on
        `holdout_mae_normalized` if you care about forecasting (requires a
        numeric cutoff); `in_sample_rmse` alone rewards the 4-parameter
        families for their extra flexibility. Always report convergence rate
        per family alongside error -- a family can look good simply by having
        failed on the hard rows, and failures are excluded from error means.
    """
    _ensure_logging_configured(log_level)

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D (n_behaviours, n_steps); got shape {data.shape}")
    n_rows, n_T = data.shape

    if t_values is None:
        t_values = np.arange(n_T, dtype=float)
    else:
        t_values = np.asarray(t_values, dtype=float)
        if t_values.shape != (n_T,):
            raise ValueError(f"t_values must have shape ({n_T},); got {t_values.shape}")
    if np.any(t_values < 0):
        raise ValueError("t_values must be non-negative (optimization steps).")
    if np.any(np.diff(t_values) <= 0):
        raise ValueError("t_values must be strictly increasing.")

    if row_ids is None:
        row_ids = [str(i) for i in range(n_rows)]
    else:
        row_ids = [str(r) for r in row_ids]
        if len(row_ids) != n_rows:
            raise ValueError(f"row_ids must have length {n_rows}; got {len(row_ids)}")

    if families is None:
        families = list(ALL_FAMILY_NAMES)
    else:
        unknown = sorted(set(families) - set(ALL_FAMILY_NAMES))
        if unknown:
            raise ValueError(f"Unknown families {unknown}; available: {sorted(ALL_FAMILY_NAMES)}")
        families = list(families)

    obs_cols = observed_columns(data)
    if len(obs_cols) == 0:
        raise ValueError("`data` contains no observed (non-NaN) values.")
    cutoffs = _resolve_cutoffs(cutoffs, obs_cols, min_fit_points)
    _warn_about_data(data, families, include_aggregate)

    if include_aggregate:
        with warnings.catch_warnings():
            # all-NaN columns (steps nobody was evaluated at) are expected.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            aggregate = np.nanmean(data, axis=0)
        all_rows = [*list(data), aggregate]
        all_row_ids = [*row_ids, "aggregate"]
    else:
        all_rows = list(data)
        all_row_ids = list(row_ids)

    tasks = [
        (rid, fam, cutoff, t_values, row, min_fit_points, random_seed, log_level)
        for rid, row in zip(all_row_ids, all_rows, strict=True)
        for fam in families
        for cutoff in cutoffs
    ]

    logger.info(
        "fit_and_forecast: %d rows (incl. aggregate=%s) x %d families x %d cutoffs = %d fits; "
        "%d/%d columns observed; n_jobs=%s",
        len(all_row_ids), include_aggregate, len(families), len(cutoffs), len(tasks),
        len(obs_cols), n_T, n_jobs,
    )

    start_wall = time.perf_counter()
    results: list[dict] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one)(*task) for task in tasks
    )
    total_wall = time.perf_counter() - start_wall
    sum_fit_wall = float(sum(r["fit_wall_time_seconds"] for r in results))

    gt_cache = {
        (r["row_id"], r["family"]): r["params"][r["asymptote_param_name"]]
        for r in results
        if r["cutoff"] == "max" and r["converged"] and r["asymptote_param_name"] is not None
    }
    for r in results:
        if r["cutoff"] == "max" or not r["converged"] or r["asymptote_param_name"] is None:
            continue
        gt = gt_cache.get((r["row_id"], r["family"]))
        if gt is None:
            continue
        estimate = r["params"][r["asymptote_param_name"]]
        abs_err = abs(estimate - gt)
        r["pseudo_ground_truth_asymptote"] = gt
        r["asymptote_abs_error"] = abs_err
        r["asymptote_rel_error"] = abs_err / abs(gt) if abs(gt) > 1e-8 else None

    df = pd.DataFrame.from_records(results, columns=list(_RESULT_COLUMNS))
    df.attrs["random_seed"] = random_seed
    df.attrs["families"] = families
    df.attrs["cutoffs"] = cutoffs
    df.attrs["total_wall_time_seconds"] = total_wall
    df.attrs["sum_fit_wall_time_seconds"] = sum_fit_wall

    n_failed = int((~df["converged"]).sum())
    if n_failed:
        by_reason = df.loc[~df["converged"], "failure_reason"].value_counts().to_dict()
        logger.info("%d/%d fits did not converge; reasons: %s", n_failed, len(df), by_reason)
    n_warned = int(df["validation_warning"].notna().sum())
    if n_warned:
        by_warning = df.loc[df["validation_warning"].notna(), "validation_warning"].value_counts().to_dict()
        logger.info(
            "%d/%d converged fits carry a soft validation warning: %s", n_warned, len(df), by_warning
        )
    logger.info(
        "fit_and_forecast complete: total wall time %.1fs, sum of per-task fit times %.1fs "
        "(%.1fx parallel speedup).",
        total_wall, sum_fit_wall, sum_fit_wall / total_wall if total_wall > 0 else float("nan"),
    )
    return df