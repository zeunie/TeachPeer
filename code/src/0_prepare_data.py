"""
Data Preparation Script

This script prepares and validates the input data for analysis.
It should be run before the main analysis pipeline.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that dataframe has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    print(f"✓ All required columns present")
    return True


def prepare_experiment1_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Prepare data for Experiment 1 analysis.
    
    Expected columns:
    - sample_id: Unique identifier for each interaction
    - student_id: Student identifier
    - week, session, idx: Temporal ordering
    - user: Student utterance
    - chatgpt_after: LLM teacher feedback
    - prev_essay: Previous version of essay
    - essay: Current version of essay
    - is_corrective_feedback: Boolean flag
    - uptake_classification: SUCCESSFUL/UNSUCCESSFUL/NO-UPTAKE
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed data
        
    Returns:
        Processed dataframe
    """
    print("\n=== Preparing Experiment 1 Data ===")
    
    # Required columns
    required_cols = [
        'sample_id', 'student_id', 'week', 'session', 'idx',
        'user', 'chatgpt_after', 'prev_essay', 'essay',
        'is_corrective_feedback'
    ]
    
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")
    
    # Validate
    if not validate_dataframe(df, required_cols):
        raise ValueError("Data validation failed!")
    
    # Sort by student and temporal order
    print("Sorting data...")
    df = df.sort_values(by=['student_id', 'week', 'session', 'idx']).reset_index(drop=True)
    
    # Filter to corrective feedback only for uptake analysis
    cf_df = df[df['is_corrective_feedback'] == True].copy()
    print(f"  {len(cf_df)} interactions with corrective feedback")
    
    # Check for missing values in key columns
    print("\nChecking for missing values:")
    for col in ['user', 'chatgpt_after', 'prev_essay']:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"  ⚠ {col}: {n_missing} missing values ({n_missing/len(df)*100:.1f}%)")
    
    # Basic statistics
    print("\nDataset statistics:")
    print(f"  Unique students: {df['student_id'].nunique()}")
    print(f"  Total interactions: {len(df)}")
    print(f"  Corrective feedback interactions: {len(cf_df)}")
    
    if 'uptake_classification' in df.columns:
        print("\nUptake distribution:")
        uptake_dist = df['uptake_classification'].value_counts()
        for label, count in uptake_dist.items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved processed data to: {output_path}")
    
    return df


def prepare_experiment2_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Prepare data for Experiment 2 analysis.
    
    Expected columns:
    - student_id: Student identifier
    - session: Session number
    - uptake_profile: low-uptake/high-uptake/mixed-uptake
    - instructional_strategy: Strategy code
    - intrinsic_load, extraneous_load, germane_load: Cognitive load ratings
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed data
        
    Returns:
        Processed dataframe
    """
    print("\n=== Preparing Experiment 2 Data ===")
    
    required_cols = [
        'student_id', 'session', 'uptake_profile'
    ]
    
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")
    
    # Validate
    if not validate_dataframe(df, required_cols):
        raise ValueError("Data validation failed!")
    
    # Check uptake profiles
    print("\nUptake profile distribution:")
    profile_dist = df['uptake_profile'].value_counts()
    for profile, count in profile_dist.items():
        print(f"  {profile}: {count} interactions")
    
    # Check if instructional_strategy column exists
    if 'instructional_strategy' in df.columns:
        print("\nInstructional strategy distribution:")
        strategy_dist = df['instructional_strategy'].value_counts()
        for strategy, count in strategy_dist.items():
            print(f"  {strategy}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  ⚠ No instructional_strategy column found")
    
    # Check cognitive load columns
    load_cols = ['intrinsic_load', 'extraneous_load', 'germane_load']
    print("\nCognitive load statistics:")
    for col in load_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {col}: M={mean_val:.2f}, SD={std_val:.2f}")
        else:
            print(f"  ⚠ {col} column not found")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved processed data to: {output_path}")
    
    return df


def create_data_summary(exp1_df: pd.DataFrame, exp2_df: pd.DataFrame = None):
    """
    Create a summary report of the datasets.
    
    Args:
        exp1_df: Experiment 1 dataframe
        exp2_df: Experiment 2 dataframe (optional)
    """
    print("\n" + "="*80)
    print("DATA SUMMARY REPORT")
    print("="*80)
    
    print("\n### Experiment 1: Real Learner vs. Baseline LLM Behavior")
    print(f"- Total interactions: {len(exp1_df)}")
    print(f"- Unique learners: {exp1_df['student_id'].nunique()}")
    print(f"- Corrective feedback interactions: {exp1_df['is_corrective_feedback'].sum()}")
    
    if exp2_df is not None:
        print("\n### Experiment 2: Learner-LLM Peer Interaction")
        print(f"- Total interactions: {len(exp2_df)}")
        print(f"- Unique learners: {exp2_df['student_id'].nunique()}")
        print(f"- Sessions per learner: {exp2_df.groupby('student_id')['session'].nunique().mean():.1f}")
    
    print("\n" + "="*80)


def main():
    """
    Main data preparation pipeline.
    """
    print("\n" + "="*80)
    print("DATA PREPARATION FOR NATURE COMMUNICATIONS SUBMISSION")
    print("="*80)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Prepare Experiment 1 data
    try:
        exp1_df = prepare_experiment1_data(
            input_path='../data/0821_simple_df.csv',
            output_path='../data/experiment_data_cf.csv'
        )
    except FileNotFoundError:
        print("\n⚠ Experiment 1 raw data not found.")
        print("Please place your data file at: ../data/0821_simple_df.csv")
        exp1_df = None
    except Exception as e:
        print(f"\n❌ Error preparing Experiment 1 data: {e}")
        exp1_df = None
    
    # Prepare Experiment 2 data
    try:
        exp2_df = prepare_experiment2_data(
            input_path='../data/experiment2_raw.csv',
            output_path='../data/experiment2_interactions.csv'
        )
    except FileNotFoundError:
        print("\n⚠ Experiment 2 raw data not found.")
        print("Please place your data file at: ../data/experiment2_raw.csv")
        exp2_df = None
    except Exception as e:
        print(f"\n❌ Error preparing Experiment 2 data: {e}")
        exp2_df = None
    
    # Create summary
    if exp1_df is not None:
        create_data_summary(exp1_df, exp2_df)
    
    print("\n✓ Data preparation complete!")
    print("\nNext step: Run the main analysis pipeline with:")
    print("  python run_all_analyses.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
