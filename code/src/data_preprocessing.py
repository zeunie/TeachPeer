"""
Data preprocessing module for LLM baseline behavior analysis.
"""
import pandas as pd
import numpy as np
from typing import Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    df = pd.read_csv(filepath)
    return df


def add_previous_essay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous essay column to each row grouped by student_id.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with prev_essay column added
    """
    # Sort by student_id, week, session, idx
    df = df.sort_values(by=["student_id", "week", "session", "idx"]).reset_index(drop=True)
    
    def add_prev_essay(group):
        group["prev_essay"] = group["essay"].shift(1)
        return group
    
    df = df.groupby("student_id", group_keys=False).apply(add_prev_essay)
    
    # Move prev_essay to the 12th column position
    cols = list(df.columns)
    prev_essay_col = cols.pop(cols.index("prev_essay"))
    insert_at = min(11, len(cols))
    cols = cols[:insert_at] + [prev_essay_col] + cols[insert_at:]
    df = df[cols]
    
    return df


def filter_corrective_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows with corrective feedback.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame containing only corrective feedback cases
    """
    if 'is_corrective_feedback' in df.columns:
        return df[df['is_corrective_feedback'] == True].copy()
    return df


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = [
        'student_id', 'week', 'session', 'idx',
        'user', 'chatgpt_after', 'prev_essay', 'essay'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def preprocess_data(filepath: str, filter_cf: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath: Path to input CSV file
        filter_cf: Whether to filter for corrective feedback only
        
    Returns:
        Preprocessed DataFrame
    """
    df = load_data(filepath)
    
    # Add previous essay if not already present
    if 'prev_essay' not in df.columns:
        df = add_previous_essay(df)
    
    # Filter for corrective feedback if requested
    if filter_cf and 'is_corrective_feedback' in df.columns:
        df = filter_corrective_feedback(df)
    
    # Validate DataFrame
    validate_dataframe(df)
    
    print(f"Preprocessing complete. Dataset shape: {df.shape}")
    print(f"Number of students: {df['student_id'].nunique()}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess data for LLM analysis")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter corrective feedback")
    
    args = parser.parse_args()
    
    df = preprocess_data(args.input, filter_cf=not args.no_filter)
    df.to_csv(args.output, index=False)
    print(f"Preprocessed data saved to {args.output}")
