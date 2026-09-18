"""
Master script to run all analyses for the paper.

This script executes the complete analysis pipeline in order:
1. Baseline LLM simulation (Experiment 1)
2. LLM peer essay generation and validation
3. Statistical analysis (Experiment 2)
"""

import sys
import os
import subprocess
from datetime import datetime


def run_script(script_path: str, description: str):
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to the script
        description: Description of the analysis
    """
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        print(f"\n✓ Completed: {description}\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {description}")
        print(f"Error output:\n{e.stderr}")
        print(f"\nStopping pipeline due to error.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


def main():
    """
    Run all analyses in sequence.
    """
    print("\n" + "="*80)
    print("NATURE COMMUNICATIONS SUBMISSION - ANALYSIS PIPELINE")
    print("Human Explanatory Behaviour with Imperfect LLM Partner")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\nPipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if config.py exists
    if not os.path.exists('config.py'):
        print("\n⚠ Warning: config.py not found!")
        print("Please copy config.example.py to config.py and add your API keys.")
        print("Continuing with default configuration...\n")
    
    # Define analysis scripts
    analyses = [
        {
            'script': 'src/1_baseline_llm_simulation.py',
            'description': 'Experiment 1: Baseline LLM Simulation'
        },
        {
            'script': 'src/2_llm_peer_generation.py',
            'description': 'LLM Peer Essay Generation & Validation'
        },
        {
            'script': 'src/3_statistical_analysis.py',
            'description': 'Experiment 2: Statistical Analysis'
        }
    ]
    
    # Run each analysis
    for i, analysis in enumerate(analyses, 1):
        print(f"\n[Step {i}/{len(analyses)}]")
        run_script(analysis['script'], analysis['description'])
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nStarted at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    
    print("\n📁 Output locations:")
    print("   - Results:  outputs/results/")
    print("   - Figures:  outputs/figures/")
    
    print("\n✓ All analyses completed successfully!")
    print("\nNext steps:")
    print("1. Review results in outputs/results/")
    print("2. Check figures in outputs/figures/")
    print("3. Verify statistical outputs match paper")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
