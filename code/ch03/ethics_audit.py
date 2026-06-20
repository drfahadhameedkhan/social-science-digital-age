"""
ethics_audit.py — Chapters 3 & 22
====================================
Research ethics tools: algorithmic fairness auditing, GDPR compliance
checking, contextual integrity evaluation, and the Chouldechova
impossibility demonstration.

Directly implements the theoretical framework from:
  - Chapter 3: Ethics in the Age of Data
  - Chapter 22: Criminology & Algorithmic Justice

Author : Fahad Hameed Khan
Book   : Social Science in the Digital Age (2025)
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# THE CHOULDECHOVA IMPOSSIBILITY  (Chapter 3, Section 3.4.2; Chapter 22.3)
# ────────────────────────────────────────────────────────────────────────────

def chouldechova_impossibility_demo(base_rate_a: float = 0.45,
                                    base_rate_b: float = 0.25,
                                    prevalence_a: float = 0.5) -> dict:
    """
    Demonstrate the Chouldechova (2017) impossibility result.

    When outcome base rates differ across groups, a risk score cannot
    simultaneously satisfy:
        (1) Calibration           — predicted risk = actual rate within group
        (2) False Positive Rate parity — equal FPR across groups
        (3) False Negative Rate parity — equal FNR across groups

    This is not a flaw in any particular algorithm — it is a mathematical
    certainty. The choice of which fairness criterion to prioritise is
    therefore a political and ethical decision, not a technical one.
    (Chapter 3, pp. XX; Chapter 22, pp. XX)

    Parameters
    ----------
    base_rate_a   : float — true recidivism rate for group A (e.g., 0.45)
    base_rate_b   : float — true recidivism rate for group B (e.g., 0.25)
    prevalence_a  : float — proportion of population that is group A

    Returns
    -------
    dict — three fairness metrics under calibration, illustrating the
           inevitable tradeoff
    """
    # Under a perfectly calibrated score, the positive predictive value (PPV)
    # is equal within each group. But if base rates differ:
    ppv_common = (
        base_rate_a * prevalence_a +
        base_rate_b * (1 - prevalence_a)
    )  # Overall PPV of a calibrated score

    # For a calibrated score, FPR and FNR must differ across groups
    # (Chouldechova 2017, Theorem 1)
    # FPR_A / FPR_B = [(1 - base_rate_A) * base_rate_B] /
    #                  [(1 - base_rate_B) * base_rate_A]
    fpr_ratio = (
        (1 - base_rate_a) * base_rate_b /
        ((1 - base_rate_b) * base_rate_a)
    )

    result = {
        "base_rate_group_A":  base_rate_a,
        "base_rate_group_B":  base_rate_b,
        "calibration_holds":  True,
        "fpr_ratio_A_to_B":   round(fpr_ratio, 3),
        "fnr_parity_holds":   False,  # BUG FIX: FNR parity CANNOT hold when calibration holds and base rates differ
        "impossibility_shown": abs(fpr_ratio - 1.0) > 0.01,
        "interpretation": (
            f"With base rates {base_rate_a} vs {base_rate_b}, "
            f"group A has {fpr_ratio:.2f}× the FPR of group B "
            f"under a perfectly calibrated algorithm. "
            f"This is mathematically inevitable — not a design choice."
        ),
    }

    print("\n" + "═" * 65)
    print("  CHOULDECHOVA (2017) IMPOSSIBILITY DEMONSTRATION")
    print("═" * 65)
    print(f"  Base rate — Group A : {base_rate_a}")
    print(f"  Base rate — Group B : {base_rate_b}")
    print(f"  Calibration holds   : {result['calibration_holds']}")
    print(f"  FNR parity holds    : {result['fnr_parity_holds']}")
    print(f"  Impossibility shown : {result['impossibility_shown']}")
    print(f"\n  → {result['interpretation']}")
    print("═" * 65)
    return result


# ────────────────────────────────────────────────────────────────────────────
# ALGORITHMIC FAIRNESS AUDITOR  (Chapters 3 & 22)
# ────────────────────────────────────────────────────────────────────────────

class FairnessAuditor:
    """
    Comprehensive algorithmic fairness audit implementing the framework
    introduced in Chapter 3 and applied in depth in Chapter 22
    (criminology / COMPAS context).

    Fairness criteria assessed:
    ─────────────────────────────────────────────────
    • Demographic Parity Difference  (DPD)
    • Equalised Odds                 (TPR gap + FPR gap)
    • Equal Opportunity              (TPR gap only)
    • Predictive Parity             (precision gap)
    • Overall accuracy disparity

    Parameters
    ----------
    y_true     : true binary labels (1 = positive outcome / recidivism)
    y_pred     : binary predictions
    protected  : pd.Series — protected attribute (race, gender, etc.)
    y_prob     : predicted probabilities (for calibration analysis)
    attr_name  : str — name of the protected attribute
    """

    def __init__(self, y_true, y_pred, protected: pd.Series,
                 y_prob=None, attr_name: str = "protected_attribute"):
        self.y_true    = np.asarray(y_true)
        self.y_pred    = np.asarray(y_pred)
        self.y_prob    = np.asarray(y_prob) if y_prob is not None else None
        self.protected = pd.Series(protected).reset_index(drop=True)
        self.attr_name = attr_name
        self.groups_   = sorted(self.protected.unique())

    def _stats(self, mask):
        yt, yp = self.y_true[mask], self.y_pred[mask]
        tp = int(((yp==1)&(yt==1)).sum())
        fp = int(((yp==1)&(yt==0)).sum())
        tn = int(((yp==0)&(yt==0)).sum())
        fn = int(((yp==0)&(yt==1)).sum())
        n  = mask.sum()
        return {
            "n":             n,
            "base_rate":     round(yt.mean(), 4),
            "accuracy":      round((tp+tn)/max(n,1), 4),
            "tpr":           round(tp/max(tp+fn,1), 4),
            "fpr":           round(fp/max(fp+tn,1), 4),
            "precision":     round(tp/max(tp+fp,1), 4),
            "positive_rate": round((tp+fp)/max(n,1), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    def group_metrics(self) -> pd.DataFrame:
        """Per-group confusion-matrix metrics."""
        records = []
        for g in self.groups_:
            mask = (self.protected == g).values
            s = self._stats(mask)
            s["group"] = g
            records.append(s)
        df = pd.DataFrame(records).set_index("group")
        return df[["n","base_rate","accuracy","tpr","fpr","precision","positive_rate"]]

    def demographic_parity_difference(self) -> float:
        """DPD = max(positive_rate) − min(positive_rate) across groups."""
        pr = [self._stats((self.protected==g).values)["positive_rate"]
              for g in self.groups_]
        return round(max(pr) - min(pr), 4)

    def equalised_odds(self) -> dict:
        """TPR gap and FPR gap across groups (Hardt et al., 2016)."""
        tprs = [self._stats((self.protected==g).values)["tpr"] for g in self.groups_]
        fprs = [self._stats((self.protected==g).values)["fpr"] for g in self.groups_]
        return {"tpr_gap": round(max(tprs)-min(tprs),4),
                "fpr_gap": round(max(fprs)-min(fprs),4)}

    def compas_style_report(self) -> dict:
        """
        Reproduce the key metrics from the ProPublica COMPAS analysis
        (Angwin et al., 2016), structured for the Chapter 22 case study.

        Returns group-level FPR, FNR, and accuracy with the
        Chouldechova impossibility flag.
        """
        metrics = self.group_metrics()
        eod     = self.equalised_odds()

        # Check base rate disparity → Chouldechova applies
        base_rates = metrics["base_rate"]
        base_rate_disparity = base_rates.max() - base_rates.min()
        chouldechova_applies = base_rate_disparity > 0.05

        report = {
            "group_metrics":         metrics,
            "fpr_gap":               eod["fpr_gap"],
            "tpr_gap":               eod["tpr_gap"],
            "base_rate_disparity":   round(base_rate_disparity, 4),
            "chouldechova_applies":  chouldechova_applies,
            "interpretation": (
                "Base rates differ significantly across groups. "
                "Per Chouldechova (2017), it is mathematically impossible "
                "for this algorithm to simultaneously satisfy calibration, "
                "FPR parity, and FNR parity. "
                "The fairness criterion chosen is a value judgement."
            ) if chouldechova_applies else (
                "Base rates are similar across groups. "
                "The Chouldechova impossibility does not apply here. "
                "Multiple fairness criteria may be simultaneously satisfiable."
            ),
        }

        self._print_compas_report(report)
        return report

    def full_audit_report(self, flag_threshold: float = 0.1) -> dict:
        """
        Full algorithmic fairness audit with flagging and recommendations.
        (Chapter 3, Section 3.4; Chapter 7 framework)
        """
        metrics = self.group_metrics()
        dpd     = self.demographic_parity_difference()
        eod     = self.equalised_odds()
        acc_gap = round(metrics["accuracy"].max()-metrics["accuracy"].min(),4)

        flags = []
        if abs(dpd) > flag_threshold:
            flags.append(f"⚠ Demographic Parity Difference = {dpd:.3f}")
        if eod["tpr_gap"] > flag_threshold:
            flags.append(f"⚠ Equal Opportunity violated — TPR gap = {eod['tpr_gap']:.3f}")
        if eod["fpr_gap"] > flag_threshold:
            flags.append(f"⚠ Equalised Odds violated — FPR gap = {eod['fpr_gap']:.3f}")
        if acc_gap > flag_threshold:
            flags.append(f"⚠ Accuracy gap across groups = {acc_gap:.3f}")

        recommendations = [
            "Apply adversarial debiasing or re-weighting during model training",
            "Consider Hardt et al. (2016) post-processing threshold adjustment",
            "Conduct intersectional analysis — bias compounds across attributes",
            "Engage affected communities in interpretation and governance",
            "Document all disparities transparently in published findings",
            "Consult Chapter 22 for the COMPAS case and Chouldechova result",
        ] if flags else ["No disparities exceed threshold — continue monitoring."]

        return {
            "attribute":              self.attr_name,
            "groups":                 self.groups_,
            "group_metrics":          metrics,
            "demographic_parity_diff": dpd,
            "equalised_odds":         eod,
            "accuracy_gap":           acc_gap,
            "flags":                  flags,
            "recommendations":        recommendations,
            "verdict": "⚠ BIAS DETECTED" if flags else "✅ WITHIN THRESHOLD",
        }

    def _print_compas_report(self, report):
        print("\n" + "═" * 65)
        print("  COMPAS-STYLE FAIRNESS ANALYSIS (Chapter 22)")
        print("═" * 65)
        print(f"\n{report['group_metrics'].round(4).to_string()}")
        print(f"\n  FPR gap (ProPublica metric) : {report['fpr_gap']}")
        print(f"  TPR gap                     : {report['tpr_gap']}")
        print(f"  Base rate disparity         : {report['base_rate_disparity']}")
        print(f"  Chouldechova applies        : {report['chouldechova_applies']}")
        print(f"\n  → {report['interpretation']}")
        print("═" * 65)


# ────────────────────────────────────────────────────────────────────────────
# CONTEXTUAL INTEGRITY CHECKER  (Chapter 3, Section 3.3)
# ────────────────────────────────────────────────────────────────────────────

class ContextualIntegrityChecker:
    """
    Implements Nissenbaum's (2010) contextual integrity framework
    for evaluating the ethics of digital data collection.

    The key question is NOT whether data is technically public,
    but whether its use matches the norms of the context in which
    it was originally produced.  (Chapter 3, Section 3.3)

    Parameters
    ----------
    data_source : str — origin platform/context
    research_use: str — proposed research use
    """

    # Contexts with their typical information flow norms
    CONTEXT_NORMS = {
        "twitter_public":       {"sensitivity": "low",    "expectation": "broadcast"},
        "facebook_public":      {"sensitivity": "medium", "expectation": "social network"},
        "reddit_public":        {"sensitivity": "varies", "expectation": "topic community"},
        "reddit_mental_health": {"sensitivity": "high",   "expectation": "peer support"},
        "facebook_group":       {"sensitivity": "high",   "expectation": "community"},
        "clinical_records":     {"sensitivity": "very_high","expectation": "treatment"},
        "government_admin":     {"sensitivity": "high",   "expectation": "administration"},
        "academic_survey":      {"sensitivity": "medium", "expectation": "research"},
        "forum_addiction":      {"sensitivity": "very_high","expectation": "peer support"},
    }

    SENSITIVITY_FLAGS = {
        "low": False, "medium": True, "high": True, "very_high": True, "varies": True
    }

    def evaluate(
        self,
        data_source: str,
        research_use: str,
        involves_linkage: bool = False,
        involves_identification: bool = False,
        involves_publication: bool = True,
    ) -> dict:
        """
        Evaluate a proposed research data use against contextual norms.

        Parameters
        ----------
        data_source             : str — key from CONTEXT_NORMS
        research_use            : str — description of proposed use
        involves_linkage        : bool — linking with other datasets
        involves_identification : bool — re-identification risk
        involves_publication    : bool — will findings be published

        Returns
        -------
        dict with ethical evaluation and recommendations
        """
        norms = self.CONTEXT_NORMS.get(data_source, {
            "sensitivity": "unknown", "expectation": "unknown"
        })
        sensitivity = norms["sensitivity"]
        requires_attention = self.SENSITIVITY_FLAGS.get(sensitivity, True)

        concerns = []
        if requires_attention:
            concerns.append(
                f"Data from '{data_source}' has {sensitivity} sensitivity. "
                f"Original context expectation: '{norms['expectation']}'. "
                f"Research use may violate contextual integrity."
            )
        if involves_linkage:
            concerns.append(
                "Data linkage substantially increases re-identification risk "
                "and may violate GDPR data minimisation principle."
            )
        if involves_identification:
            concerns.append(
                "Re-identification risk requires DPIA under GDPR Article 35 "
                "and IRB/REC ethical review."
            )
        if involves_publication and sensitivity in ("high", "very_high"):
            concerns.append(
                "Publishing findings from high-sensitivity contexts requires "
                "full anonymisation and may require participant consent "
                "beyond IRB approval."
            )

        return {
            "data_source":    data_source,
            "sensitivity":    sensitivity,
            "research_use":   research_use,
            "concerns":       concerns,
            "requires_irb":   requires_attention or involves_linkage,
            "requires_dpia":  involves_identification or involves_linkage,
            "recommendation": (
                "Proceed with full IRB review, DPIA, and explicit consent plan."
                if concerns else
                "Lower risk — standard IRB review recommended."
            ),
        }


# ────────────────────────────────────────────────────────────────────────────
# GDPR SPECIAL CATEGORY CHECKER  (Chapter 3, Section 3.6)
# ────────────────────────────────────────────────────────────────────────────

GDPR_SPECIAL_CATEGORIES = {
    "racial_ethnic_origin":    "Article 9(1)(a) — requires explicit consent or substantial public interest",
    "political_opinions":      "Article 9(1)(b)",
    "religious_beliefs":       "Article 9(1)(c)",
    "trade_union_membership":  "Article 9(1)(d)",
    "genetic_data":            "Article 9(1)(e)",
    "biometric_data":          "Article 9(1)(f)",
    "health_data":             "Article 9(1)(g) — most common in social science",
    "sex_life_orientation":    "Article 9(1)(h)",
    "criminal_convictions":    "Article 10 — separate provision, strict requirements",
}


def check_gdpr_categories(variables: List[str]) -> dict:
    """
    Check whether a list of study variables involves GDPR special categories.

    Parameters
    ----------
    variables : list of variable/column names or data types collected

    Returns
    -------
    dict with special category flags and compliance requirements
    """
    flags = []
    for var in variables:
        var_lower = var.lower()
        for cat, requirement in GDPR_SPECIAL_CATEGORIES.items():
            keywords = cat.replace("_", " ").split()
            if any(kw in var_lower for kw in keywords):
                flags.append({"variable": var, "category": cat,
                               "requirement": requirement})
    return {
        "variables_checked": variables,
        "special_categories_found": len(flags) > 0,
        "flags": flags,
        "compliance_actions": [
            "Identify lawful basis under Article 6",
            "Identify separate condition under Article 9 for each special category",
            "Conduct DPIA (Article 35) — high-risk processing",
            "Implement pseudonymisation and encryption",
            "Limit data retention to minimum necessary",
            "Obtain explicit ethics committee approval",
        ] if flags else ["No special categories detected — standard GDPR compliance applies."],
    }
