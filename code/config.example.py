"""
Configuration file template for API keys and parameters.

Copy this file to config.py and fill in your API keys.
"""

# OpenAI API Configuration
OPENAI_API_KEY = "your-openai-api-key-here"

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o"
CLASSIFICATION_MODEL = "gpt-4o"

# Rate Limiting
MAX_CONCURRENT_REQUESTS = 10

# Statistical Analysis Parameters
N_CLUSTERS = 3
RANDOM_SEED = 42
EQUIVALENCE_DELTA_MULTIPLIER = 0.5  # For TOST: ±0.5 × SD

# Bootstrap Parameters
N_BOOTSTRAP_SAMPLES = 5000
CONFIDENCE_LEVEL = 0.95

# Figure Settings
DPI = 300
FIGURE_FORMAT = "pdf"
