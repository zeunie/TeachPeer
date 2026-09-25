"""
Experiment 1 clustering and Experiment 2 statistical analyses.

Runs k-means clustering, chi-square tests, ICAP post-hoc tests, ANOVAs,
and the R binomial GLMMs.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CODE_ROOT, DATA_DIR, RESULT_DIR
from utils import cramers_v

STRATEGIES = [
    "Amplification",
    "Reformulate",
    "Accept",
    "Reject",
    "Exemplification",
    "Connection",
    "Metacognitive",
    "Praise",
]
ICAP_LEVELS = ["Constructive", "Active", "Passive"]
GROUPS = ["low-uptake", "high-uptake", "mixed-uptake"]
COMPARISONS = [
    ("low-uptake", "high-uptake"),
    ("low-uptake", "mixed-uptake"),
    ("high-uptake", "mixed-uptake"),
]
LOAD_VARS = ["intrinsic_load", "extraneous_load", "germane_load"]


def prepare_uptake_profiles(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for student_id, student_data in df.groupby("student_id"):
        cf_data = student_data[student_data["is_corrective_feedback"] == True]
        if len(cf_data) == 0:
            continue
        counts = cf_data["uptake_classification"].value_counts()
        total = len(cf_data)
        records.append(
            {
                "student_id": student_id,
                "no_uptake": counts.get("NO-UPTAKE", 0) / total,
                "successful": counts.get("SUCCESSFUL", 0) / total,
                "unsuccessful": counts.get("UNSUCCESSFUL", 0) / total,
                "total_interactions": total,
            }
        )
    return pd.DataFrame(records)


def cluster_uptake_profiles(profiles_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    feature_cols = ["no_uptake", "successful", "unsuccessful"]
    X = profiles_df[feature_cols].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    profiles_df = profiles_df.copy()
    profiles_df["cluster"] = kmeans.fit_predict(X)

    representatives = []
    for cluster_id in range(n_clusters):
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]
        features = cluster_data[feature_cols].values
        distances = np.sqrt(((features - kmeans.cluster_centers_[cluster_id]) ** 2).sum(axis=1))
        nearest = distances.argmin()
        representatives.append(
            {
                "no_uptake": features[nearest, 0],
                "successful": features[nearest, 1],
                "unsuccessful": features[nearest, 2],
                "cluster": cluster_id,
                "representative_student_id": cluster_data.iloc[nearest]["student_id"],
            }
        )
    centroids_df = pd.DataFrame(representatives)
    cluster_labels = {
        centroids_df["no_uptake"].idxmax(): "low-uptake",
        centroids_df["successful"].idxmax(): "high-uptake",
    }
    remaining = [i for i in range(n_clusters) if i not in cluster_labels]
    if remaining:
        cluster_labels[remaining[0]] = "mixed-uptake"
    profiles_df["uptake_profile"] = profiles_df["cluster"].map(cluster_labels)
    centroids_df["uptake_profile"] = centroids_df["cluster"].map(cluster_labels)
    return profiles_df, centroids_df


def binary_chisquare(df: pd.DataFrame, column: str, levels: list) -> pd.DataFrame:
    rows = []
    for level in levels:
        indicator = (df[column] == level).astype(int)
        contingency = pd.crosstab(indicator, df["group_type"])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        rows.append(
            {
                column: level,
                "chi2": chi2,
                "df": dof,
                "p_value": p_value,
                "n": len(df),
                "cramers_v": cramers_v(chi2, len(df), *contingency.shape),
            }
        )
        print(f"  {level}: chi2({dof}) = {chi2:.2f}, p = {p_value:.4g}, V = {rows[-1]['cramers_v']:.3f}")
    return pd.DataFrame(rows)


def icap_posthoc(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for level in ICAP_LEVELS:
        indicator = (df["icap"] == level).astype(int)
        print(f"\n{level}")
        for group1, group2 in COMPARISONS:
            g1 = indicator[df["group_type"] == group1]
            g2 = indicator[df["group_type"] == group2]
            p1, p2 = g1.mean(), g2.mean()
            p_pool = (g1.sum() + g2.sum()) / (len(g1) + len(g2))
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(g1) + 1 / len(g2)))
            z_stat = (p1 - p2) / se if se > 0 else 0.0
            p_raw = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            p_adj = min(p_raw * len(COMPARISONS), 1.0)
            print(f"  {group1} vs {group2}: z = {z_stat:.3f}, p_adj = {p_adj:.4g}")
            results.append(
                {
                    "icap_level": level,
                    "comparison": f"{group1} vs {group2}",
                    "group1": group1,
                    "group2": group2,
                    "prop1": p1,
                    "prop2": p2,
                    "n1_level": int(g1.sum()),
                    "n1_total": len(g1),
                    "n2_level": int(g2.sum()),
                    "n2_total": len(g2),
                    "z_statistic": z_stat,
                    "p_raw": p_raw,
                    "p_bonferroni": p_adj,
                    "significant": p_adj < 0.05,
                }
            )
    return pd.DataFrame(results)


def eta_squared(groups: list) -> float:
    all_data = pd.concat(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_data - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def run_anova(df: pd.DataFrame, column: str) -> dict:
    groups = [df.loc[df["group_type"] == group, column].dropna() for group in GROUPS]
    f_stat, p_value = stats.f_oneway(*groups)
    levene_stat, levene_p = stats.levene(*groups)
    result = {
        "variable": column,
        "f_statistic": float(f_stat),
        "df_between": 2,
        "df_within": sum(len(g) for g in groups) - 3,
        "p_value": float(p_value),
        "eta_squared": eta_squared(groups),
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "low_mean": float(groups[0].mean()),
        "low_sd": float(groups[0].std()),
        "high_mean": float(groups[1].mean()),
        "high_sd": float(groups[1].std()),
        "mixed_mean": float(groups[2].mean()),
        "mixed_sd": float(groups[2].std()),
    }
    print(f"  {column}: F(2, {result['df_within']}) = {f_stat:.3f}, p = {p_value:.4g}, eta^2 = {result['eta_squared']:.3f}")
    if p_value < 0.05:
        for i, j in ((0, 1), (0, 2), (1, 2)):
            t_stat, p_raw = stats.ttest_ind(groups[i], groups[j])
            print(f"    {GROUPS[i]} vs {GROUPS[j]}: t = {t_stat:.3f}, p_adj = {min(p_raw * 3, 1.0):.4g}")
    return result


def run_glmm() -> bool:
    script = CODE_ROOT / "src" / "run_glmm_in_R.R"
    try:
        subprocess.run(["Rscript", str(script)], check=True, cwd=CODE_ROOT)
        return True
    except FileNotFoundError:
        print("Rscript not found. Skipping GLMMs. Install R with lme4 and dplyr to run them.")
        return False
    except subprocess.CalledProcessError:
        print("GLMM script failed.")
        return False


def main() -> None:
    print("Experiment 1: uptake-profile clustering")
    exp1 = pd.read_csv(DATA_DIR / "experiment_1.csv")
    exp1["student_id"] = exp1["sample_id"].astype(str).str.split("-").str[0]
    profiles_df, centroids_df = cluster_uptake_profiles(prepare_uptake_profiles(exp1))
    profiles_df.to_csv(RESULT_DIR / "uptake_profiles_clustered.csv", index=False)
    centroids_df.to_csv(RESULT_DIR / "cluster_centroids.csv", index=False)
    print(centroids_df[["uptake_profile", "no_uptake", "successful", "unsuccessful"]].to_string(index=False))

    print("\nExperiment 2: instructional-strategy and ICAP chi-square tests")
    exp2 = pd.read_csv(DATA_DIR / "experiment_2_chat.csv")
    user_df = exp2[exp2["role"] == "user"].copy()
    icap_df = user_df[user_df["icap"].isin(ICAP_LEVELS)].copy()
    print(f"  User utterances: {len(user_df)}; ICAP utterances: {len(icap_df)}")
    binary_chisquare(user_df, "strategy", STRATEGIES).to_csv(RESULT_DIR / "strategy_chisquare.csv", index=False)
    binary_chisquare(icap_df, "icap", ICAP_LEVELS).rename(columns={"icap": "icap_level"}).to_csv(
        RESULT_DIR / "icap_chisquare.csv", index=False
    )

    print("\nICAP pairwise proportion tests")
    icap_posthoc(icap_df).to_csv(RESULT_DIR / "icap_posthoc_bonferroni.csv", index=False)

    print("\nCognitive load and essay improvement")
    session_df = pd.read_csv(DATA_DIR / "cognitive_load_data.csv")
    participant_df = session_df.groupby(["participant_id", "group_type"], as_index=False).agg(
        {
            "intrinsic_load": "mean",
            "extraneous_load": "mean",
            "germane_load": "mean",
            "essay_improvement": "first",
        }
    )
    pd.DataFrame([run_anova(participant_df, column) for column in LOAD_VARS]).to_csv(
        RESULT_DIR / "cognitive_load_anova.csv", index=False
    )
    essay_df = participant_df[participant_df["essay_improvement"].notna()]
    pd.DataFrame([run_anova(essay_df, "essay_improvement")]).to_csv(
        RESULT_DIR / "essay_improvement_anova.csv", index=False
    )

    print("\nBinomial GLMMs")
    run_glmm()
    print(f"\nWrote results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
