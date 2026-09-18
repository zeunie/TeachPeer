"""
Statistical Analysis for Experiment 2

This script:
1. Performs k-means clustering on learner uptake profiles
2. Analyzes instructional strategies across uptake profiles
3. Maps strategies to ICAP engagement levels
4. Conducts chi-square tests and mixed-effects models
5. Analyzes cognitive load across conditions
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def prepare_uptake_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare learner uptake profiles from interaction data.
    
    Args:
        df: Dataframe with learner interactions
        
    Returns:
        Dataframe with uptake proportions per learner
    """
    # Group by learner and calculate uptake proportions
    learner_profiles = []
    
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]
        
        # Only consider interactions with corrective feedback
        cf_data = student_data[student_data['is_corrective_feedback'] == True]
        
        if len(cf_data) == 0:
            continue
        
        uptake_counts = cf_data['uptake_classification'].value_counts()
        total = len(cf_data)
        
        learner_profiles.append({
            'student_id': student_id,
            'no_uptake': uptake_counts.get('NO-UPTAKE', 0) / total,
            'successful': uptake_counts.get('SUCCESSFUL', 0) / total,
            'unsuccessful': uptake_counts.get('UNSUCCESSFUL', 0) / total,
            'total_interactions': total
        })
    
    return pd.DataFrame(learner_profiles)


def cluster_uptake_profiles(
    profiles_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster learners based on their uptake profiles using k-means.
    
    Args:
        profiles_df: Dataframe with uptake profiles
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (profiles with cluster labels, fitted KMeans model)
    """
    # Features for clustering
    X = profiles_df[['no_uptake', 'successful', 'unsuccessful']].values
    
    # Standardize features (optional, but can help with visualization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    profiles_df['cluster'] = kmeans.fit_predict(X)
    
    # Calculate cluster centroids in original space
    centroids = []
    for i in range(n_clusters):
        cluster_data = profiles_df[profiles_df['cluster'] == i]
        centroids.append({
            'cluster': i,
            'no_uptake': cluster_data['no_uptake'].mean(),
            'successful': cluster_data['successful'].mean(),
            'unsuccessful': cluster_data['unsuccessful'].mean(),
            'size': len(cluster_data)
        })
    
    centroids_df = pd.DataFrame(centroids)
    
    # Label clusters based on dominant behavior
    # Low-uptake: high no-uptake
    # High-uptake: high successful
    # Mixed-uptake: mixed patterns
    for i, row in centroids_df.iterrows():
        if row['no_uptake'] > 0.7:
            label = 'low-uptake'
        elif row['successful'] > 0.5:
            label = 'high-uptake'
        else:
            label = 'mixed-uptake'
        centroids_df.loc[i, 'label'] = label
        profiles_df.loc[profiles_df['cluster'] == i, 'cluster_label'] = label
    
    print("\n=== Cluster Centroids ===")
    print(centroids_df)
    
    return profiles_df, kmeans, centroids_df


def plot_uptake_clusters(
    profiles_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    output_path: str = '../outputs/figures/uptake_clusters.pdf'
):
    """
    Create 3D visualization of uptake clusters.
    
    Args:
        profiles_df: Dataframe with cluster assignments
        centroids_df: Cluster centroids
        output_path: Path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    colors = ['red', 'blue', 'green']
    for i, row in centroids_df.iterrows():
        cluster_data = profiles_df[profiles_df['cluster'] == i]
        ax1.scatter(
            cluster_data['no_uptake'],
            cluster_data['successful'],
            cluster_data['unsuccessful'],
            c=colors[i],
            label=row['label'],
            alpha=0.6
        )
    
    ax1.set_xlabel('NO-UPTAKE')
    ax1.set_ylabel('SUCCESSFUL')
    ax1.set_zlabel('UNSUCCESSFUL')
    ax1.set_title('3D Uptake Profile Clusters')
    ax1.legend()
    
    # 2D projection: NO-UPTAKE vs SUCCESSFUL
    ax2 = fig.add_subplot(132)
    for i, row in centroids_df.iterrows():
        cluster_data = profiles_df[profiles_df['cluster'] == i]
        ax2.scatter(
            cluster_data['no_uptake'],
            cluster_data['successful'],
            c=colors[i],
            label=row['label'],
            alpha=0.6
        )
    
    ax2.set_xlabel('NO-UPTAKE')
    ax2.set_ylabel('SUCCESSFUL')
    ax2.set_title('NO-UPTAKE vs SUCCESSFUL')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 2D projection: NO-UPTAKE vs UNSUCCESSFUL
    ax3 = fig.add_subplot(133)
    for i, row in centroids_df.iterrows():
        cluster_data = profiles_df[profiles_df['cluster'] == i]
        ax3.scatter(
            cluster_data['no_uptake'],
            cluster_data['unsuccessful'],
            c=colors[i],
            label=row['label'],
            alpha=0.6
        )
    
    ax3.set_xlabel('NO-UPTAKE')
    ax3.set_ylabel('UNSUCCESSFUL')
    ax3.set_title('NO-UPTAKE vs UNSUCCESSFUL')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cluster visualization saved to {output_path}")


def analyze_instructional_strategies(
    df: pd.DataFrame,
    strategy_col: str = 'instructional_strategy'
) -> pd.DataFrame:
    """
    Analyze distribution of instructional strategies by uptake profile.
    
    Args:
        df: Dataframe with coded instructional strategies
        strategy_col: Column name for strategy codes
        
    Returns:
        Dataframe with strategy proportions by profile
    """
    # Group by uptake profile
    profiles = df['uptake_profile'].unique()
    
    results = []
    for profile in profiles:
        profile_data = df[df['uptake_profile'] == profile]
        
        strategy_counts = profile_data[strategy_col].value_counts()
        total = len(profile_data)
        
        for strategy, count in strategy_counts.items():
            results.append({
                'profile': profile,
                'strategy': strategy,
                'count': count,
                'proportion': count / total
            })
    
    results_df = pd.DataFrame(results)
    
    # Pivot for chi-square test
    contingency = results_df.pivot_table(
        index='strategy',
        columns='profile',
        values='count',
        fill_value=0
    )
    
    # Chi-square test for each strategy
    print("\n=== Chi-square Tests for Instructional Strategies ===")
    for strategy in contingency.index:
        row = contingency.loc[strategy].values.reshape(1, -1)
        total_row = contingency.sum(axis=0).values.reshape(1, -1) - row
        table = np.vstack([row, total_row])
        
        chi2, p, dof, expected = stats.chi2_contingency(table)
        print(f"{strategy}: χ²({dof}) = {chi2:.2f}, p = {p:.4f}")
    
    return results_df


def map_to_icap(strategy: str) -> str:
    """
    Map instructional strategy to ICAP engagement level.
    
    Args:
        strategy: Strategy code
        
    Returns:
        ICAP level (Constructive, Active, or Passive)
    """
    strategy_mapping = {
        'Amplification': 'Constructive',
        'Metacognitive': 'Constructive',
        'Connection': 'Constructive',
        'Reformulate': 'Active',
        'Exemplification': 'Active',
        'Reject': 'Active',
        'Accept': 'Passive',
        'Praise': 'Passive'
    }
    return strategy_mapping.get(strategy, 'Active')


def analyze_icap_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ICAP engagement levels by uptake profile.
    
    Args:
        df: Dataframe with strategy codes
        
    Returns:
        Dataframe with ICAP proportions by profile
    """
    # Map strategies to ICAP levels
    df['icap_level'] = df['instructional_strategy'].apply(map_to_icap)
    
    # Analyze by profile
    profiles = df['uptake_profile'].unique()
    
    results = []
    for profile in profiles:
        profile_data = df[df['uptake_profile'] == profile]
        
        icap_counts = profile_data['icap_level'].value_counts()
        total = len(profile_data)
        
        for level, count in icap_counts.items():
            results.append({
                'profile': profile,
                'icap_level': level,
                'count': count,
                'proportion': count / total
            })
    
    results_df = pd.DataFrame(results)
    
    # Chi-square tests
    print("\n=== Chi-square Tests for ICAP Engagement ===")
    for level in ['Constructive', 'Active', 'Passive']:
        level_data = results_df[results_df['icap_level'] == level]
        contingency = level_data.pivot_table(
            index='icap_level',
            columns='profile',
            values='count',
            fill_value=0
        ).values
        
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"{level}: χ²({dof}) = {chi2:.2f}, p = {p:.4f}")
    
    return results_df


def plot_strategy_distribution(
    results_df: pd.DataFrame,
    output_path: str = '../outputs/figures/strategy_distribution.pdf'
):
    """
    Create stacked bar plot of instructional strategies by profile.
    
    Args:
        results_df: Dataframe with strategy proportions
        output_path: Path to save figure
    """
    # Pivot data
    pivot_df = results_df.pivot_table(
        index='profile',
        columns='strategy',
        values='proportion',
        fill_value=0
    )
    
    # Sort profiles
    profile_order = ['low-uptake', 'high-uptake', 'mixed-uptake']
    pivot_df = pivot_df.reindex(profile_order)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_df.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        colormap='tab10',
        width=0.7
    )
    
    ax.set_xlabel('Proportion (%)')
    ax.set_ylabel('Uptake Profile')
    ax.set_title('Distribution of Instructional Strategies by Uptake Profile')
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, 1)
    
    # Format x-axis as percentage
    ax.set_xticklabels([f'{int(x*100)}' for x in ax.get_xticks()])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Strategy distribution plot saved to {output_path}")


def plot_icap_engagement(
    results_df: pd.DataFrame,
    output_path: str = '../outputs/figures/icap_engagement.pdf'
):
    """
    Create grouped bar plot of ICAP engagement levels by profile.
    
    Args:
        results_df: Dataframe with ICAP proportions
        output_path: Path to save figure
    """
    # Pivot data
    pivot_df = results_df.pivot_table(
        index='profile',
        columns='icap_level',
        values='proportion',
        fill_value=0
    )
    
    # Sort profiles
    profile_order = ['low-uptake', 'high-uptake', 'mixed-uptake']
    pivot_df = pivot_df.reindex(profile_order)
    
    # Reorder ICAP levels
    icap_order = ['Constructive', 'Active', 'Passive']
    pivot_df = pivot_df[icap_order]
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_df.plot(
        kind='barh',
        ax=ax,
        color=['#2ecc71', '#3498db', '#95a5a6'],
        width=0.7
    )
    
    ax.set_xlabel('Proportion (%)')
    ax.set_ylabel('Uptake Profile')
    ax.set_title('ICAP Engagement Levels by Uptake Profile')
    ax.legend(title='ICAP Level')
    ax.set_xlim(0, 1)
    
    # Format x-axis as percentage
    ax.set_xticklabels([f'{int(x*100)}' for x in ax.get_xticks()])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ICAP engagement plot saved to {output_path}")


def analyze_cognitive_load(df: pd.DataFrame) -> Dict:
    """
    Analyze cognitive load across uptake profiles using ANOVA.
    
    Args:
        df: Dataframe with cognitive load ratings
        
    Returns:
        Dictionary with ANOVA results
    """
    profiles = df['uptake_profile'].unique()
    
    results = {}
    
    for load_type in ['intrinsic_load', 'extraneous_load', 'germane_load']:
        print(f"\n=== {load_type.replace('_', ' ').title()} ===")
        
        # Prepare data for each profile
        groups = [df[df['uptake_profile'] == p][load_type].dropna() for p in profiles]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        print(f"F({len(groups)-1}, {sum(len(g) for g in groups) - len(groups)}) = {f_stat:.3f}, p = {p_value:.4f}")
        
        # Post-hoc pairwise comparisons (Bonferroni correction)
        if p_value < 0.05:
            print("Significant! Running post-hoc tests...")
            
            from itertools import combinations
            n_comparisons = len(list(combinations(range(len(profiles)), 2)))
            alpha_corrected = 0.05 / n_comparisons
            
            for i, j in combinations(range(len(profiles)), 2):
                t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                sig = "***" if p_val < alpha_corrected else ""
                print(f"  {profiles[i]} vs {profiles[j]}: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")
        
        results[load_type] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'means': {p: df[df['uptake_profile'] == p][load_type].mean() for p in profiles}
        }
    
    return results


def main():
    """
    Main execution function.
    """
    print("=== Statistical Analysis: Experiment 2 ===\n")
    
    # Load Experiment 1 data for clustering
    print("Loading Experiment 1 data...")
    exp1_df = pd.read_csv('../data/experiment_data_cf.csv')
    
    # Prepare uptake profiles
    print("Preparing uptake profiles...")
    profiles_df = prepare_uptake_profiles(exp1_df)
    
    # Cluster uptake profiles
    print("\nClustering uptake profiles...")
    profiles_df, kmeans, centroids_df = cluster_uptake_profiles(profiles_df, n_clusters=3)
    
    # Save clustering results
    profiles_df.to_csv('../outputs/results/uptake_profiles_clustered.csv', index=False)
    centroids_df.to_csv('../outputs/results/cluster_centroids.csv', index=False)
    
    # Visualize clusters
    print("\nCreating cluster visualization...")
    plot_uptake_clusters(profiles_df, centroids_df)
    
    # Load Experiment 2 data
    print("\nLoading Experiment 2 data...")
    exp2_df = pd.read_csv('../data/experiment2_interactions.csv')
    
    # Analyze instructional strategies
    print("\nAnalyzing instructional strategies...")
    strategy_results = analyze_instructional_strategies(exp2_df)
    strategy_results.to_csv('../outputs/results/strategy_analysis.csv', index=False)
    
    # Analyze ICAP engagement
    print("\nAnalyzing ICAP engagement...")
    icap_results = analyze_icap_engagement(exp2_df)
    icap_results.to_csv('../outputs/results/icap_analysis.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_strategy_distribution(strategy_results)
    plot_icap_engagement(icap_results)
    
    # Analyze cognitive load
    print("\nAnalyzing cognitive load...")
    load_results = analyze_cognitive_load(exp2_df)
    
    # Save load results
    pd.DataFrame(load_results).to_csv('../outputs/results/cognitive_load_analysis.csv')
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
