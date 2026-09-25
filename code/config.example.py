"""
Configuration template for optional LLM generation.

Copy this file to config.py only if you want to regenerate model outputs:
    cp config.example.py config.py

The default analysis pipeline does not need API keys. Table 1 labels and
peer essays are already included in the released dataset.
"""

OPENAI_API_KEY = "your-openai-api-key-here"

DEFAULT_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o"
CLASSIFICATION_MODEL = "gpt-4o"
STUDENT_MODEL = "gpt-4o-mini"
DIFF_CHECK_MODEL = "gpt-4o"

TEMPERATURE = 0.7
MAX_COMPLETION_TOKENS = 1000
MAX_CONCURRENT_REQUESTS = 10

UPTAKE_LABELS = ["SUCCESSFUL", "UNSUCCESSFUL", "NO-UPTAKE"]
N_CLUSTERS = 3
RANDOM_SEED = 42
EQUIVALENCE_DELTA_MULTIPLIER = 0.5
N_BOOTSTRAP_SAMPLES = 5000
CONFIDENCE_LEVEL = 0.95

DPI = 300
FIGURE_FORMAT = "pdf"
