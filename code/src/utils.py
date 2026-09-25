"""Shared statistical and text-feature utilities."""

import ast
import json
import re
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

UPTAKE_LABELS = ["SUCCESSFUL", "NO-UPTAKE", "UNSUCCESSFUL"]


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    min_dim = min(n_rows - 1, n_cols - 1)
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / min_dim))


def normalize_uptake(value):
    text = str(value).strip()
    if text in UPTAKE_LABELS:
        return text
    return pd.NA


def tost_equivalence_test(group1: np.ndarray, group2: np.ndarray, delta: float) -> Dict[str, float]:
    diff = float(np.mean(group1) - np.mean(group2))
    se = float(np.sqrt(np.var(group1, ddof=1) / len(group1) + np.var(group2, ddof=1) / len(group2)))
    df = len(group1) + len(group2) - 2
    t_upper = (diff - delta) / se
    t_lower = (diff + delta) / se
    p_tost = float(max(stats.t.cdf(t_upper, df=df), 1 - stats.t.cdf(t_lower, df=df)))
    t_crit = stats.t.ppf(0.975, df)
    return {
        "mean_diff": diff,
        "se": se,
        "ci_lower": diff - t_crit * se,
        "ci_upper": diff + t_crit * se,
        "p_tost": p_tost,
        "delta": delta,
    }


def kolmogorov_smirnov_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    result = stats.ks_2samp(group1, group2)
    return float(result.statistic), float(result.pvalue)


def wasserstein_distance_normalized(group1: np.ndarray, group2: np.ndarray) -> float:
    w_dist = stats.wasserstein_distance(group1, group2)
    pooled_sd = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    return float(w_dist / pooled_sd) if pooled_sd > 0 else 0.0


def jensen_shannon_divergence(group1: np.ndarray, group2: np.ndarray, bins: int = 30) -> float:
    min_val = min(group1.min(), group2.min())
    max_val = max(group1.max(), group2.max())
    p, _ = np.histogram(group1, bins=bins, range=(min_val, max_val), density=True)
    q, _ = np.histogram(group2, bins=bins, range=(min_val, max_val), density=True)
    p = p / p.sum() + 1e-10
    q = q / q.sum() + 1e-10
    m = (p + q) / 2
    return float((stats.entropy(p, m) + stats.entropy(q, m)) / 2)


def range_overlap_proportion(reference: np.ndarray, test: np.ndarray) -> float:
    within = ((test >= reference.min()) & (test <= reference.max())).sum()
    return float(within / len(test))


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    stats_boot = [statistic_func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1 - confidence
    return float(np.percentile(stats_boot, 100 * alpha / 2)), float(np.percentile(stats_boot, 100 * (1 - alpha / 2)))


def parse_kc_list(value) -> List[dict]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(text)))


def type_token_ratio(text: str) -> float:
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", str(text))]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _syllable_count(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", str(text))
    sentences = [s for s in re.split(r"[.!?]+", str(text)) if s.strip()]
    if not words or not sentences:
        return 0.0
    syllables = sum(_syllable_count(word) for word in words)
    return float(206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words)))
