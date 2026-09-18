"""
LLM Peer Essay Generation with Error Mirroring

This script:
1. Extracts error patterns (KC errors) from learner essays
2. Generates LLM peer essays with mirrored error patterns
3. Validates LLM peer essays using:
   - Equivalence testing (TOST)
   - Distribution comparison (KS test, Wasserstein distance)
   - Range validation
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from typing import Dict, List, Tuple
from utils import (
    tost_equivalence_test,
    kolmogorov_smirnov_test,
    wasserstein_distance_normalized,
    jensen_shannon_divergence,
    bootstrap_ci,
    range_overlap_proportion
)


# Configuration
OPENAI_API_KEY = "your-api-key-here"
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_knowledge_components(essay: str, model: str = "gpt-4o") -> Dict:
    """
    Extract incorrect knowledge components from essay.
    
    Args:
        essay: Student essay text
        model: Model to use for extraction
        
    Returns:
        Dictionary with KC errors categorized by type
    """
    prompt = f"""
Analyze the following essay and identify errors in three categories:

1. Content errors: Incorrect facts, weak arguments, logical fallacies
2. Organization errors: Poor structure, weak transitions, unclear flow
3. Language errors: Grammar, spelling, word choice, punctuation

For each error, provide:
- Error type (content/organization/language)
- Description of the error
- Severity (low/medium/high)

Essay:
{essay}

Return your analysis in JSON format:
{{
  "content_errors": [...],
  "organization_errors": [...],
  "language_errors": [...]
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=1024
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        # Count total KC errors
        total_kc = (
            len(result.get('content_errors', [])) +
            len(result.get('organization_errors', [])) +
            len(result.get('language_errors', []))
        )
        
        result['total_kc_count'] = total_kc
        return result
        
    except Exception as e:
        print(f"Error extracting KCs: {e}")
        return {'total_kc_count': 0}


def generate_peer_essay_with_errors(
    topic: str,
    target_word_count: int,
    error_profile: Dict,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a peer essay with specified error patterns.
    
    Args:
        topic: Essay topic
        target_word_count: Target length
        error_profile: Dictionary of errors to inject
        model: Model to use
        
    Returns:
        Generated essay with errors
    """
    prompt = f"""
Write an essay on the topic: {topic}

Requirements:
- Target length: approximately {target_word_count} words
- Include the following types of errors:
  * Content errors: {len(error_profile.get('content_errors', []))} errors
  * Organization errors: {len(error_profile.get('organization_errors', []))} errors
  * Language errors: {len(error_profile.get('language_errors', []))} errors

Write a student-level essay that naturally incorporates these error types.
The essay should sound like it was written by a learner, not an expert.

Essay:
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=target_word_count * 2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating essay: {e}")
        return ""


def calculate_essay_features(essay: str) -> Dict:
    """
    Calculate linguistic and structural features of an essay.
    
    Returns:
        Dictionary with word count, TTR, and FRES
    """
    words = essay.split()
    word_count = len(words)
    
    # Type-Token Ratio
    unique_words = len(set(words))
    ttr = (unique_words / word_count * 100) if word_count > 0 else 0
    
    # Flesch Reading Ease Score (simplified approximation)
    # For accurate FRES, use textstat library
    sentences = essay.split('.')
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    
    avg_sentence_length = word_count / sentence_count
    
    # Simplified syllable count (rough approximation)
    syllable_count = sum([len([c for c in word if c.lower() in 'aeiou']) for word in words])
    avg_syllables_per_word = syllable_count / max(1, word_count)
    
    fres = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    
    return {
        'word_count': word_count,
        'ttr': ttr,
        'fres': fres
    }


def run_equivalence_testing(
    human_values: np.ndarray,
    llm_values: np.ndarray,
    feature_name: str,
    delta_multiplier: float = 0.5
) -> Dict:
    """
    Run TOST equivalence test for a given feature.
    
    Args:
        human_values: Human learner values
        llm_values: LLM peer values
        feature_name: Name of the feature
        delta_multiplier: Multiplier for equivalence margin (default 0.5 SD)
        
    Returns:
        Dictionary with test results
    """
    # Calculate equivalence margin: ±0.5 × SD of human distribution
    delta = delta_multiplier * np.std(human_values, ddof=1)
    
    # Run TOST
    tost_results = tost_equivalence_test(llm_values, human_values, delta)
    
    # KS test
    ks_stat, ks_pvalue = kolmogorov_smirnov_test(human_values, llm_values)
    
    # Wasserstein distance
    w_dist = wasserstein_distance_normalized(human_values, llm_values)
    
    # JS divergence
    js_div = jensen_shannon_divergence(human_values, llm_values)
    
    # Range overlap
    range_overlap = range_overlap_proportion(human_values, llm_values)
    
    return {
        'feature': feature_name,
        'delta': delta,
        'mean_diff': tost_results['mean_diff'],
        'ci_lower': tost_results['ci_lower'],
        'ci_upper': tost_results['ci_upper'],
        'p_tost': tost_results['p_tost'],
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'wasserstein_distance': w_dist,
        'js_divergence': js_div,
        'range_overlap': range_overlap,
        'equivalence_achieved': (tost_results['p_tost'] < 0.05)
    }


def plot_distribution_comparison(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    features: List[str],
    output_path: str = '../outputs/figures/distribution_comparison.pdf'
):
    """
    Create distribution comparison plots (box plots + density plots).
    
    Args:
        human_df: Human learner dataframe
        llm_df: LLM peer dataframe
        features: List of features to plot
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(len(features), 2, figsize=(12, 4*len(features)))
    
    feature_labels = {
        'kc_count': 'Number of Incorrect KC',
        'word_count': 'Word Count',
        'ttr': 'Type Token Ratio (%)',
        'fres': 'Flesch Reading Ease Score'
    }
    
    for i, feature in enumerate(features):
        # Box plot
        ax_box = axes[i, 0] if len(features) > 1 else axes[0]
        data_to_plot = [
            human_df[feature].dropna(),
            llm_df[feature].dropna()
        ]
        bp = ax_box.boxplot(data_to_plot, labels=['Human Learner', 'LLM Partner'],
                            patch_artist=True)
        bp['boxes'][0].set_facecolor('orange')
        bp['boxes'][1].set_facecolor('green')
        ax_box.set_ylabel(feature_labels.get(feature, feature))
        ax_box.set_title('Box Plot')
        ax_box.grid(alpha=0.3)
        
        # Density plot
        ax_density = axes[i, 1] if len(features) > 1 else axes[1]
        
        human_data = human_df[feature].dropna()
        llm_data = llm_df[feature].dropna()
        
        # Plot densities
        ax_density.hist(human_data, bins=30, alpha=0.5, color='orange', 
                       label='Human Learner', density=True)
        ax_density.hist(llm_data, bins=30, alpha=0.5, color='green',
                       label='LLM Partner', density=True)
        
        # Add range boundaries
        human_min, human_max = human_data.min(), human_data.max()
        ax_density.axvline(human_min, color='gray', linestyle='--', alpha=0.5)
        ax_density.axvline(human_max, color='gray', linestyle='--', alpha=0.5)
        ax_density.axvspan(human_min, human_max, alpha=0.1, color='gray')
        
        ax_density.set_xlabel(feature_labels.get(feature, feature))
        ax_density.set_ylabel('Density')
        ax_density.set_title('Density Plot')
        ax_density.legend()
        ax_density.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison plot saved to {output_path}")


def plot_tost_results(
    results_df: pd.DataFrame,
    output_path: str = '../outputs/figures/tost_equivalence.pdf'
):
    """
    Create TOST equivalence test visualization.
    
    Args:
        results_df: Dataframe with TOST results
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = results_df['feature'].tolist()
    y_positions = np.arange(len(features))
    
    for i, row in results_df.iterrows():
        # Determine color based on equivalence
        color = 'green' if row['equivalence_achieved'] else 'orange'
        
        # Plot CI
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
               color=color, linewidth=2, marker='|', markersize=10)
        
        # Plot mean difference
        ax.plot(row['mean_diff'], i, 'o', color=color, markersize=8)
        
        # Plot equivalence bounds
        ax.axvline(-row['delta'], color='gray', linestyle='--', alpha=0.5)
        ax.axvline(row['delta'], color='gray', linestyle='--', alpha=0.5)
        
        # Shade equivalence region
        ax.axvspan(-row['delta'], row['delta'], alpha=0.1, color='green')
    
    # Zero reference line
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features)
    ax.set_xlabel('Mean Difference (LLM − Human)')
    ax.set_title('Two One-Sided Tests (TOST) for Practical Equivalence')
    ax.grid(alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='Equivalent'),
        Patch(facecolor='orange', alpha=0.5, label='Not Equivalent')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TOST results plot saved to {output_path}")


def main():
    """
    Main execution function.
    """
    print("=== LLM Peer Essay Generation & Validation ===\n")
    
    # Load learner data
    print("Loading learner data...")
    df = pd.read_csv('../data/experiment_data_cf.csv')
    
    # Extract features from human essays
    print("Extracting features from human essays...")
    human_features = []
    for i, row in df.iterrows():
        essay = row['prev_essay']
        kc_info = extract_knowledge_components(essay)
        essay_features = calculate_essay_features(essay)
        
        human_features.append({
            'kc_count': kc_info['total_kc_count'],
            'word_count': essay_features['word_count'],
            'ttr': essay_features['ttr'],
            'fres': essay_features['fres']
        })
    
    human_df = pd.DataFrame(human_features)
    
    # Generate LLM peer essays with error mirroring
    print("\nGenerating LLM peer essays with error mirroring...")
    llm_features = []
    topics = ["The impact of social media", "Climate change solutions", 
              "The future of education", "Work-life balance"]
    
    for i, row in df.iterrows():
        essay = row['prev_essay']
        kc_info = extract_knowledge_components(essay)
        essay_features = calculate_essay_features(essay)
        
        # Generate peer essay with mirrored errors
        topic = topics[i % len(topics)]
        peer_essay = generate_peer_essay_with_errors(
            topic,
            essay_features['word_count'],
            kc_info
        )
        
        # Calculate features of generated essay
        peer_features = calculate_essay_features(peer_essay)
        peer_kc = extract_knowledge_components(peer_essay)
        
        llm_features.append({
            'kc_count': peer_kc['total_kc_count'],
            'word_count': peer_features['word_count'],
            'ttr': peer_features['ttr'],
            'fres': peer_features['fres'],
            'essay': peer_essay
        })
    
    llm_df = pd.DataFrame(llm_features)
    
    # Save generated essays
    llm_df.to_csv('../outputs/results/llm_peer_essays.csv', index=False)
    print("LLM peer essays saved.")
    
    # Run equivalence testing
    print("\n=== Running Equivalence Testing ===\n")
    features_to_test = ['kc_count', 'word_count', 'ttr', 'fres']
    results = []
    
    for feature in features_to_test:
        print(f"Testing {feature}...")
        result = run_equivalence_testing(
            human_df[feature].values,
            llm_df[feature].values,
            feature
        )
        results.append(result)
        
        print(f"  Mean difference: {result['mean_diff']:.2f}")
        print(f"  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
        print(f"  TOST p-value: {result['p_tost']:.4f}")
        print(f"  Equivalence achieved: {result['equivalence_achieved']}")
        print(f"  KS test: D={result['ks_statistic']:.2f}, p={result['ks_pvalue']:.4f}")
        print(f"  Wasserstein distance: {result['wasserstein_distance']:.2f} SD")
        print(f"  Range overlap: {result['range_overlap']*100:.1f}%\n")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('../outputs/results/equivalence_test_results.csv', index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_distribution_comparison(human_df, llm_df, features_to_test)
    plot_tost_results(results_df)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
