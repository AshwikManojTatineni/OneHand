"""Statistics fixed by protocol.yaml: Wilson intervals, paired McNemar and Wilcoxon tests, Holm correction, retention."""
from math import comb, sqrt

import numpy as np
from scipy import stats


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(a_success, b_success):
    """Exact two-sided McNemar test on paired binary outcomes (same seeds and layouts)."""
    a, b = np.asarray(a_success, bool), np.asarray(b_success, bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def wilcoxon_stage(a_stage, b_stage):
    a, b = np.asarray(a_stage), np.asarray(b_stage)
    if np.all(a == b):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def retention(sr_target, sr_source):
    return float("nan") if sr_source == 0 else sr_target / sr_source


def stage_retention(stages_target, stages_source):
    s = np.mean(stages_source)
    return float("nan") if s == 0 else float(np.mean(stages_target) / s)
