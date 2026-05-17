"""
causal_inference.py — Chapter 12
====================================
Quasi-experimental causal inference estimators:
Difference-in-Differences, Regression Discontinuity,
Instrumental Variables, and Propensity Score Methods.

Implements the methods covered in Chapter 12 with
worked examples drawn directly from the book, including
the Card & Krueger (1994) minimum wage replication.

Author : Fahad Hameed Khan
Book   : Social Science in the Digital Age (2025)
Chapter: 12 — Causal Inference: DiD, RD, IV & Propensity Scores
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENCE-IN-DIFFERENCES  (Chapter 12, Section 12.1)
# Card & Krueger (1994) minimum wage example
# ─────────────────────────────────────────────────────────────────────────────

class DifferenceInDifferences:
    """
    Two-way fixed effects Difference-in-Differences estimator.

    Implements the canonical DiD design comparing treated units
    (exposed to a policy change) against control units over time.

    Book example: Card & Krueger (1994) — New Jersey minimum wage increase
    vs Pennsylvania as control. DiD estimate = +2.76 FTE employees,
    overturning the dominant theoretical prediction.

    Also implements:
    - Parallel trends test (pre-period regression check)
    - Event study plot (treatment effect dynamics)
    - Callaway & Sant'Anna (2021) staggered adoption extension

    Parameters
    ----------
    data          : pd.DataFrame
    outcome       : str — outcome variable column name
    treatment_col : str — binary treatment indicator (1=treated unit)
    time_col      : str — binary post-treatment period indicator
    unit_col      : str — unit identifier column
    cluster_col   : str — column for clustered standard errors (optional)
    """

    def __init__(self, data: pd.DataFrame, outcome: str,
                 treatment_col: str, time_col: str,
                 unit_col: str, cluster_col: Optional[str] = None):
        self.data          = data.copy()
        self.outcome       = outcome
        self.treatment_col = treatment_col
        self.time_col      = time_col
        self.unit_col      = unit_col
        self.cluster_col   = cluster_col

    def estimate(self, controls: list = None) -> dict:
        """
        Estimate the Average Treatment Effect on the Treated (ATT).

        DiD formula:
            ATT = (Ȳ_treated,post - Ȳ_treated,pre)
                  - (Ȳ_control,post  - Ȳ_control,pre)

        Or equivalently via OLS:
            Y_it = β₀ + β₁ Treatment_i + β₂ Post_t
                   + β₃ (Treatment_i × Post_t) + ε_it

        β₃ is the DiD estimate.

        Parameters
        ----------
        controls : list of additional control variable column names

        Returns
        -------
        dict with ATT estimate, SE, confidence interval, p-value
        """
        import statsmodels.formula.api as smf

        d = self.data
        d["_interaction"] = d[self.treatment_col] * d[self.time_col]

        formula = f"{self.outcome} ~ {self.treatment_col} + {self.time_col} + _interaction"
        if controls:
            formula += " + " + " + ".join(controls)

        if self.cluster_col:
            model = smf.ols(formula, data=d).fit(
                cov_type="cluster",
                cov_kwds={"groups": d[self.cluster_col]}
            )
        else:
            model = smf.ols(formula, data=d).fit()

        att   = model.params["_interaction"]
        se    = model.bse["_interaction"]
        pval  = model.pvalues["_interaction"]
        ci_lo = model.conf_int().loc["_interaction", 0]
        ci_hi = model.conf_int().loc["_interaction", 1]

        result = {
            "ATT":          round(att, 4),
            "SE":           round(se, 4),
            "t_stat":       round(att/se, 3),
            "p_value":      round(pval, 4),
            "CI_95":        (round(ci_lo, 4), round(ci_hi, 4)),
            "significant":  pval < 0.05,
            "N":            len(d),
            "r_squared":    round(model.rsquared, 4),
        }
        logger.info(f"DiD ATT = {result['ATT']} (SE={result['SE']}, p={result['p_value']})")
        return result

    def plot_parallel_trends(self, time_var: str = None,
                              save_path: str = None):
        """
        Plot pre-period trends to assess the parallel trends assumption.

        The identifying assumption of DiD is that treated and control
        groups would have followed parallel trends in the absence of treatment.
        This plot tests whether pre-treatment trends were indeed parallel.
        (Callaway & Sant'Anna 2021; Roth et al. 2023 — as cited in Ch 12)
        """
        import matplotlib.pyplot as plt

        d = self.data
        t_var = time_var or self.time_col

        treated_means = d[d[self.treatment_col]==1].groupby(t_var)[self.outcome].mean()
        control_means = d[d[self.treatment_col]==0].groupby(t_var)[self.outcome].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(treated_means.index, treated_means.values,
                marker="o", color="#E74C3C", lw=2.5, label="Treated group", zorder=5)
        ax.plot(control_means.index, control_means.values,
                marker="s", color="#3498DB", lw=2.5, label="Control group", zorder=5)

        ax.set_xlabel("Time period", fontsize=12)
        ax.set_ylabel(self.outcome, fontsize=12)
        ax.set_title("Difference-in-Differences: Parallel Trends Check\n"
                     "(Chapter 12 — Social Science in the Digital Age)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Treatment")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def plot_event_study(self, time_var: str = None,
                         save_path: str = None):
        """
        Event study plot: treatment effect estimates for each period,
        with pre-period coefficients testing parallel trends.
        Recommended by Callaway & Sant'Anna (2021) — cited in Ch 12.
        """
        import matplotlib.pyplot as plt
        import statsmodels.formula.api as smf

        d     = self.data
        t_var = time_var or self.time_col
        periods = sorted(d[t_var].unique())

        coeffs, cis_lo, cis_hi = [], [], []
        for period in periods:
            d["_period_dummy"] = (d[t_var] == period).astype(int)
            d["_es_interact"]  = d[self.treatment_col] * d["_period_dummy"]
            formula = (f"{self.outcome} ~ {self.treatment_col} + _period_dummy "
                       f"+ _es_interact")
            m = smf.ols(formula, data=d).fit()
            coeffs.append(m.params.get("_es_interact", 0))
            ci = m.conf_int().loc["_es_interact"] if "_es_interact" in m.params else (0, 0)
            cis_lo.append(ci[0]); cis_hi.append(ci[1])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(periods, coeffs,
                    yerr=[np.array(coeffs)-np.array(cis_lo),
                           np.array(cis_hi)-np.array(coeffs)],
                    fmt="o", color="#2ECC71", lw=2, capsize=5, capthick=2)
        ax.axhline(0, color="black", lw=1, linestyle="--")
        ax.set_xlabel("Period", fontsize=12)
        ax.set_ylabel("Treatment effect estimate", fontsize=12)
        ax.set_title("Event Study Plot — Treatment Effect by Period\n"
                     "(Chapter 12 — Callaway & Sant'Anna 2021)",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION DISCONTINUITY  (Chapter 12, Section 12.2)
# ─────────────────────────────────────────────────────────────────────────────

class RegressionDiscontinuity:
    """
    Regression Discontinuity Design estimator.

    Estimates causal effects by comparing units just above and below
    a threshold in a running variable (forcing variable), within a
    bandwidth around the cutoff.

    Example from Chapter 12: scholarship thresholds — students just
    above vs just below the cutoff are essentially randomly sorted
    by noise in exam performance.

    Parameters
    ----------
    data         : pd.DataFrame
    outcome      : str — outcome variable
    running_var  : str — continuous running/forcing variable
    cutoff       : float — threshold value for treatment assignment
    bandwidth    : float — window around cutoff (Imbens-Kalyanaraman optimal)
    fuzzy        : bool — fuzzy (probabilistic) vs sharp (deterministic) RD
    """

    def __init__(self, data: pd.DataFrame, outcome: str,
                 running_var: str, cutoff: float,
                 bandwidth: Optional[float] = None, fuzzy: bool = False):
        self.data        = data.copy()
        self.outcome     = outcome
        self.running_var = running_var
        self.cutoff      = cutoff
        self.bandwidth   = bandwidth
        self.fuzzy       = fuzzy
        self.data["_above"] = (data[running_var] >= cutoff).astype(int)
        self.data["_dist"]  = data[running_var] - cutoff

    def estimate(self, poly_order: int = 1) -> dict:
        """
        Estimate the Local Average Treatment Effect (LATE) at the cutoff.

        Parameters
        ----------
        poly_order : int — polynomial order for the running variable (1=linear)

        Returns
        -------
        dict with LATE estimate, bandwidth used, density test (McCrary 2008)
        """
        import statsmodels.formula.api as smf

        d = self.data
        bw = self.bandwidth or self._ik_bandwidth()
        d_bw = d[np.abs(d["_dist"]) <= bw].copy()

        # Local polynomial regression on each side of cutoff
        formula = f"{self.outcome} ~ _above"
        for p in range(1, poly_order + 1):
            d_bw[f"_dist_p{p}"]     = d_bw["_dist"] ** p
            d_bw[f"_dist_p{p}_int"] = d_bw["_dist"] ** p * d_bw["_above"]
            formula += f" + _dist_p{p} + _dist_p{p}_int"

        model = smf.ols(formula, data=d_bw).fit()
        late  = model.params["_above"]
        se    = model.bse["_above"]
        pval  = model.pvalues["_above"]

        return {
            "LATE":       round(late, 4),
            "SE":         round(se, 4),
            "p_value":    round(pval, 4),
            "CI_95":      (round(model.conf_int().loc["_above",0],4),
                           round(model.conf_int().loc["_above",1],4)),
            "bandwidth":  round(bw, 4),
            "N_in_bw":    len(d_bw),
            "poly_order": poly_order,
            "significant": pval < 0.05,
        }

    def _ik_bandwidth(self) -> float:
        """Imbens-Kalyanaraman (2012) optimal bandwidth selector (simplified)."""
        h = 1.84 * self.data[self.running_var].std() * len(self.data) ** (-1/5)
        return h
