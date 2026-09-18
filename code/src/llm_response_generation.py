"""
LLM response generation module using GPT-4o-mini to simulate student responses.
"""
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


async def generate_student_response_async(
    prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    client: AsyncOpenAI
) -> Tuple[int, Optional[str]]:
    """
    Generate student response asynchronously using GPT-4o-mini.
    
    Args:
        prompt: Input prompt for the model
        semaphore: Asyncio semaphore for rate limiting
        index: Index of the current row
        client: AsyncOpenAI client
        
    Returns:
        Tuple of (index, response_text)
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


def create_student_response_prompt(
    student_essay: str,
    student_utterance: str,
    teacher_feedback: str
) -> str:
    """
    Create prompt for generating student's next response.
    
    Args:
        student_essay: Student's written essay
        student_utterance: Student's current utterance
        teacher_feedback: Teacher's corrective feedback
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a student focusing on English writing. Analyze your written essay carefully, explain your thought process (1024 tokens or less), and try to apply the concepts you've learned to revise the essay. If you're unsure, express your uncertainty and explain your reasoning.

Your written Essay: [{student_essay}]

You: [{student_utterance}]

Feedback provider: [{teacher_feedback}]

Your next response here:"""
    
    return prompt


async def process_all_responses(df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    """
    Process all rows to generate student responses.
    
    Args:
        df: Input DataFrame
        client: AsyncOpenAI client
        
    Returns:
        DataFrame with generated responses added
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    tasks = []
    
    for i in range(len(df)):
        student_essay = df.loc[i, "prev_essay"]
        student_utterance = df.loc[i, "user"]
        teacher_feedback = df.loc[i, "chatgpt_after"]
        
        prompt = create_student_response_prompt(
            student_essay, student_utterance, teacher_feedback
        )
        
        task = generate_student_response_async(prompt, semaphore, i, client)
        tasks.append(task)
    
    # Execute with progress bar
    results = []
    for coro in tqdm.as_completed(tasks, desc="Generating student responses", total=len(tasks)):
        result = await coro
        results.append(result)
    
    # Save results
    df["llm_next_response"] = None
    for idx, response in results:
        if response:
            df.loc[idx, "llm_next_response"] = response.strip()
    
    return df


async def main_async(input_path: str, output_path: str):
    """
    Main async function for response generation.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    # Initialize async client
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Generate responses
    print(f"Generating responses for {len(df)} rows...")
    df = await process_all_responses(df, client)
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print statistics
    non_null_responses = df["llm_next_response"].notna().sum()
    print(f"Generated {non_null_responses}/{len(df)} responses successfully")


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
    
    parser = argparse.ArgumentParser(description="Generate LLM student responses")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
