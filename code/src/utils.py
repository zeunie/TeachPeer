"""
Utility functions for statistical analysis and data processing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """
    Calculate Cramér's V statistic for effect size.
    
    Args:
        confusion_matrix: Contingency table as numpy array
        
    Returns:
        Cramér's V value (0 to 1)
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


def tost_equivalence_test(
    group1: np.ndarray,
    group2: np.ndarray,
    delta: float
) -> Dict[str, float]:
    """
    Perform Two One-Sided Tests (TOST) for equivalence.
    
    Args:
        group1: First group data
        group2: Second group data
        delta: Equivalence margin (±delta)
        
    Returns:
        Dictionary containing test statistics
    """
    diff = np.mean(group1) - np.mean(group2)
    se = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
    
    # Test if difference < upper bound
    t_upper = (diff - delta) / se
    p_upper = stats.t.cdf(t_upper, df=len(group1) + len(group2) - 2)
    
    # Test if difference > lower bound
    t_lower = (diff + delta) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=len(group1) + len(group2) - 2)
    
    # TOST p-value is the maximum of the two one-sided tests
    p_tost = max(p_upper, p_lower)
    
    # Confidence interval
    df = len(group1) + len(group2) - 2
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
    
    return {
        'mean_diff': diff,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_tost': p_tost,
        'delta': delta
    }


def kolmogorov_smirnov_test(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distribution comparison.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Tuple of (D statistic, p-value)
    """
    return stats.ks_2samp(group1, group2)


def wasserstein_distance_normalized(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    Calculate normalized Wasserstein distance.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Normalized Wasserstein distance in SD units
    """
    w_dist = stats.wasserstein_distance(group1, group2)
    pooled_sd = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    return w_dist / pooled_sd if pooled_sd > 0 else 0


def jensen_shannon_divergence(
    group1: np.ndarray,
    group2: np.ndarray,
    bins: int = 30
) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    Args:
        group1: First group data
        group2: Second group data
        bins: Number of bins for histogram
        
    Returns:
        JS divergence value
    """
    # Create histograms
    min_val = min(group1.min(), group2.min())
    max_val = max(group1.max(), group2.max())
    
    p, _ = np.histogram(group1, bins=bins, range=(min_val, max_val), density=True)
    q, _ = np.histogram(group2, bins=bins, range=(min_val, max_val), density=True)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    
    # Calculate JS divergence
    m = (p + q) / 2
    js_div = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
    
    return js_div


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 5000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data
        statistic_func: Function to calculate statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower bound, upper bound)
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return lower, upper


def range_overlap_proportion(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    Calculate proportion of group2 values falling within group1 range.
    
    Args:
        group1: Reference group (typically human learners)
        group2: Test group (typically LLM peers)
        
    Returns:
        Proportion of group2 values within group1 range
    """
    min_val = group1.min()
    max_val = group1.max()
    
    within_range = ((group2 >= min_val) & (group2 <= max_val)).sum()
    return within_range / len(group2)
