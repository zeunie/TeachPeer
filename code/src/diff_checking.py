"""
Diff checking module to identify changes between original and revised essays.
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


async def check_diff_async(
    prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    client: AsyncOpenAI
) -> Tuple[int, Optional[str]]:
    """
    Check differences between essays asynchronously.
    
    Args:
        prompt: Input prompt for the model
        semaphore: Asyncio semaphore for rate limiting
        index: Index of the current row
        client: AsyncOpenAI client
        
    Returns:
        Tuple of (index, diff_result)
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


def create_diff_prompt(essay_before: str, essay_after: str) -> str:
    """
    Create prompt for diff checking.
    
    Args:
        essay_before: Original essay
        essay_after: Revised essay
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a text comparison tool. Your task is to find the differences between two essays — "essay_before" and "essay_after". You should focus on identifying sentences that have been added, removed, or modified.

For each modified sentence, provide both the original and the updated version in the following format:
- "type": "modified"
- "text": "<original sentence> -> <updated sentence>"

For added sentences:
- "type": "added"
- "text": "<added sentence>"

For removed sentences:
- "type": "removed"
- "text": "<removed sentence>"

Here is an example:

essay_before:
In the famous Korean movie "parasite", there are some people who do not worry about economic problem, but there are some people who have trouble living their lives due to their poor situation. Is it reasonable for the wealthy people to be helped by the government with the people who live in an underground floor? I think it is not desirable. The government should focus on helping only those most in need.

essay_after:
In the famous Korean movie "parasite", there are some people who do not worry about economic problem, but there are some people who have trouble living their lives due to their poor situation. Is it reasonable for the wealthy people to be helped by the government with the people who live in an underground floor? It is obviously not desirable. The government should focus on helping only those most in need.

Expected output:
{{
  "diff": [
    {{
      "type": "modified",
      "text": "I think it is not desirable -> It is obviously not desirable"
    }}
  ]
}}

Please provide the differences in the format above, without any extra explanation.

essay_before = {essay_before}
essay_after = {essay_after}

output="""
    
    return prompt


async def process_all_diffs(df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    """
    Process all rows to check essay differences.
    
    Args:
        df: Input DataFrame with essays
        client: AsyncOpenAI client
        
    Returns:
        DataFrame with diff results and edit flags added
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    tasks = []
    
    # Initialize column
    df["llm_sentence_diff"] = None
    
    for i in range(len(df)):
        essay_before = df.loc[i, 'prev_essay']
        essay_after = df.loc[i, 'llm_essay_generated']
        
        # Skip if essay wasn't generated
        if pd.isna(essay_after):
            df.loc[i, 'llm_sentence_diff'] = 'No essay generated'
            continue
        
        # If essays are identical, mark as no difference
        if essay_before == essay_after:
            df.loc[i, 'llm_sentence_diff'] = 'No difference'
        else:
            prompt = create_diff_prompt(essay_before, essay_after)
            task = check_diff_async(prompt, semaphore, i, client)
            tasks.append(task)
    
    # Execute with progress bar
    results = []
    for coro in tqdm.as_completed(tasks, desc="Checking differences", total=len(tasks)):
        result = await coro
        results.append(result)
    
    # Save results
    for idx, diff_result in results:
        if diff_result:
            df.loc[idx, 'llm_sentence_diff'] = diff_result.strip()
    
    # Add is_essay_edited flag
    df['llm_is_essay_edited'] = False
    for i in range(len(df)):
        if (pd.notna(df.loc[i, 'llm_sentence_diff']) and
            df.loc[i, 'llm_sentence_diff'].strip() != 'No difference' and
            df.loc[i, 'llm_sentence_diff'].strip() != 'No essay generated' and
            df.loc[i, 'llm_sentence_diff'].strip() != '{"diff": []}'):
            df.loc[i, 'llm_is_essay_edited'] = True
    
    return df


async def main_async(input_path: str, output_path: str):
    """
    Main async function for diff checking.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    # Initialize async client
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check for required column
    if "llm_essay_generated" not in df.columns:
        raise ValueError("Input data must contain 'llm_essay_generated' column. "
                       "Run essay_generation.py first.")
    
    # Check diffs
    print(f"Checking differences for {len(df)} rows...")
    df = await process_all_diffs(df, client)
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print statistics
    edited_count = df['llm_is_essay_edited'].sum()
    print(f"Essays edited: {edited_count}/{len(df)} ({edited_count/len(df)*100:.1f}%)")


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
    
    parser = argparse.ArgumentParser(description="Check essay differences")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
