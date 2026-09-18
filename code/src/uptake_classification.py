"""
Uptake classification module to classify learner uptake after corrective feedback.
"""
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


async def classify_uptake_async(
    prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    client: AsyncOpenAI
) -> Tuple[int, Optional[str]]:
    """
    Classify uptake asynchronously.
    
    Args:
        prompt: Input prompt for the model
        semaphore: Asyncio semaphore for rate limiting
        index: Index of the current row
        client: AsyncOpenAI client
        
    Returns:
        Tuple of (index, classification_label)
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=config.DIFF_CHECK_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS
            )
            return (index, response.choices[0].message.content)
        except Exception as e:
            print(f"Error at index {index}: {e}")
            return (index, None)


def create_uptake_classification_prompt(
    corrective_feedback: str,
    revision_record: str
) -> str:
    """
    Create prompt for uptake classification.
    
    Args:
        corrective_feedback: Teacher's corrective feedback
        revision_record: Record of learner's revisions (diff)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are classifying learner uptake after corrective feedback.

Definitions:
- SUCCESSFUL: The learner corrected the error(s) after receiving feedback, and no further repair is needed.
- NO-UPTAKE: The learner ignored the feedback entirely, did not revise based on it, and made no engagement with the suggested changes in the feedback.
- UNSUCCESSFUL: The learner tried to correct the error(s) using the feedback but failed to make sufficient improvements, requiring further revision.

Task:
Based on the given data, classify the learner uptake into exactly one of: SUCCESSFUL, NO-UPTAKE, UNSUCCESSFUL.

Data:
corrective_feedback: {corrective_feedback}
learner's revision record: {revision_record}

Answer with only one label."""
    
    return prompt


async def process_all_classifications(df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    """
    Process all rows to classify uptake.
    
    Args:
        df: Input DataFrame with diff results
        client: AsyncOpenAI client
        
    Returns:
        DataFrame with uptake classifications added
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    tasks = []
    
    # Initialize column
    df["llm_uptake_classification"] = "NO-UPTAKE"
    
    for i in range(len(df)):
        # Only classify if essay was edited and feedback is corrective
        if (df.loc[i, 'llm_is_essay_edited'] == True and
            df.loc[i, 'is_corrective_feedback'] == True):
            
            corrective_feedback = df.loc[i, 'chatgpt_after']
            revision_record = df.loc[i, 'llm_sentence_diff']
            
            prompt = create_uptake_classification_prompt(
                corrective_feedback, revision_record
            )
            
            task = classify_uptake_async(prompt, semaphore, i, client)
            tasks.append(task)
    
    # Execute with progress bar
    results = []
    for coro in tqdm.as_completed(tasks, desc="Classifying uptake", total=len(tasks)):
        result = await coro
        results.append(result)
    
    # Save results
    for idx, classification in results:
        if classification:
            df.loc[idx, 'llm_uptake_classification'] = classification.strip()
    
    return df


async def main_async(input_path: str, output_path: str):
    """
    Main async function for uptake classification.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    # Initialize async client
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check for required columns
    required_cols = ['llm_sentence_diff', 'llm_is_essay_edited', 'is_corrective_feedback']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data must contain columns: {missing_cols}. "
                       "Run diff_checking.py first.")
    
    # Classify uptake
    print(f"Classifying uptake for {len(df)} rows...")
    df = await process_all_classifications(df, client)
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print statistics
    print("\nUptake Classification Results:")
    print(df['llm_uptake_classification'].value_counts())
    print("\nPercentages:")
    for label in config.UPTAKE_LABELS:
        count = (df['llm_uptake_classification'] == label).sum()
        percentage = count / len(df) * 100
        print(f"{label}: {count}/{len(df)} ({percentage:.2f}%)")


def main(input_path: str, output_path: str):
    """
    Main function wrapper for async execution.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    asyncio.run(main_async(input_path, output_path))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify learner uptake")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
