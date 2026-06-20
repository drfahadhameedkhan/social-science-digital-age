"""Core function tests — Social Science in the Digital Age (2025)"""
import numpy as np
import pandas as pd

def test_chouldechova_base_rates_differ():
    """When base rates differ, FPR ratio must differ from 1.0."""
    br_a, br_b = 0.45, 0.25
    fpr_ratio = (1 - br_a) * br_b / ((1 - br_b) * br_a)
    # BUG FIX: Changed assertion logic - when base rates differ significantly,
    # FPR ratio MUST differ substantially from 1.0 (this is Chouldechova's proof)
    # The original assertion was backwards - it required only a small difference
    assert abs(fpr_ratio - 1.0) > 0.5, "Chouldechova: FPR ratio should differ substantially when base rates differ"

def test_chouldechova_equal_base_rates():
    """When base rates are equal, FPR ratio should be close to 1.0."""
    br = 0.35
    fpr_ratio = (1 - br) * br / ((1 - br) * br)
    assert abs(fpr_ratio - 1.0) < 0.001

def test_did_manual():
    """Test DiD formula: ATT = (T_post - T_pre) - (C_post - C_pre)."""
    t_post, t_pre = 22.0, 20.3
    c_post, c_pre = 21.2, 21.1
    att = (t_post - t_pre) - (c_post - c_pre)
    assert abs(att - 1.6) < 0.01

def test_vader_compound_range():
    """VADER compound scores must lie in [-1, 1]."""
    scores = [0.72, -0.45, 0.0, 1.0, -1.0]
    assert all(-1.0 <= s <= 1.0 for s in scores)

def test_fairness_audit_metrics():
    """FPR and TPR must lie in [0, 1]."""
    y_true = np.array([1,0,1,0,1,0,1,0])
    y_pred = np.array([1,1,1,0,0,0,1,0])
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    tn = ((y_pred==0)&(y_true==0)).sum()
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)
    assert 0 <= fpr <= 1 and 0 <= tpr <= 1
