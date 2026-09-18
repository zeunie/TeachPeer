"""
Essay generation module using GPT-4o-mini to generate revised essays.
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


async def generate_revised_essay_async(
    prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    client: AsyncOpenAI
) -> Tuple[int, Optional[str]]:
    """
    Generate revised essay asynchronously using GPT-4o-mini.
    
    Args:
        prompt: Input prompt for the model
        semaphore: Asyncio semaphore for rate limiting
        index: Index of the current row
        client: AsyncOpenAI client
        
    Returns:
        Tuple of (index, essay_text)
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=config.STUDENT_MODEL,
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


def create_essay_revision_prompt(
    student_utterance: str,
    teacher_feedback: str,
    student_next_response: str,
    previous_essay: str
) -> str:
    """
    Create prompt for generating revised essay.
    
    Args:
        student_utterance: Student's initial utterance
        teacher_feedback: Teacher's feedback
        student_next_response: Student's response to feedback
        previous_essay: Previous version of the essay
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Based on your current understanding after the conversation, rethink step by step and then revise the essay.

You: [{student_utterance}]

Feedback provider: [{teacher_feedback}]

You: [{student_next_response}]

Your previous essay: [{previous_essay}]

Revised essay:"""
    
    return prompt


async def process_all_essays(df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    """
    Process all rows to generate revised essays.
    
    Args:
        df: Input DataFrame with responses
        client: AsyncOpenAI client
        
    Returns:
        DataFrame with generated essays added
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    tasks = []
    
    for i in range(len(df)):
        student_utterance = df.loc[i, "user"]
        teacher_feedback = df.loc[i, "chatgpt_after"]
        student_next_response = df.loc[i, "llm_next_response"]
        previous_essay = df.loc[i, "prev_essay"]
        
        # Skip if response is missing
        if pd.isna(student_next_response):
            continue
        
        prompt = create_essay_revision_prompt(
            student_utterance, teacher_feedback, 
            student_next_response, previous_essay
        )
        
        task = generate_revised_essay_async(prompt, semaphore, i, client)
        tasks.append(task)
    
    # Execute with progress bar
    results = []
    for coro in tqdm.as_completed(tasks, desc="Generating revised essays", total=len(tasks)):
        result = await coro
        results.append(result)
    
    # Save results
    df["llm_essay_generated"] = None
    for idx, essay in results:
        if essay:
            df.loc[idx, "llm_essay_generated"] = essay.strip()
    
    return df


async def main_async(input_path: str, output_path: str):
    """
    Main async function for essay generation.
    
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
    if "llm_next_response" not in df.columns:
        raise ValueError("Input data must contain 'llm_next_response' column. "
                       "Run llm_response_generation.py first.")
    
    # Generate essays
    print(f"Generating essays for {len(df)} rows...")
    df = await process_all_essays(df, client)
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print statistics
    non_null_essays = df["llm_essay_generated"].notna().sum()
    print(f"Generated {non_null_essays}/{len(df)} essays successfully")


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
    
    parser = argparse.ArgumentParser(description="Generate revised essays")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
