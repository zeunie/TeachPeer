"""
Configuration file for API keys and model settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model Settings
STUDENT_MODEL = "gpt-4o-mini"  # Model for student responses
DIFF_CHECK_MODEL = "gpt-4o-2024-08-06"  # Model for diff checking

# API Settings
MAX_COMPLETION_TOKENS = 1024
TEMPERATURE = 0
MAX_CONCURRENT_REQUESTS = 10  # Semaphore limit for async requests

# File Paths
DEFAULT_INPUT_PATH = "data/experiment_data_cf.csv"
DEFAULT_OUTPUT_DIR = "outputs"

# Uptake Classification Labels
UPTAKE_LABELS = ["SUCCESSFUL", "NO-UPTAKE", "UNSUCCESSFUL"]
