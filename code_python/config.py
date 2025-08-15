from pathlib import Path
from math import erf, sqrt

# Paths
PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR.parent / "data"
PLOTS_DIR = PKG_DIR.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Orders
MODEL_ORDER = [
    "Claude v1.3",
    "Claude v2",
    "Claude 3 Sonnet",
    "Claude 3.5 Sonnet",
    "GPT-3.5 Turbo",
    "GPT-4",
    "GPT-4o",
    "Llama 3 8B",
    "Llama 3 70B",
    "Mistral 7B",
    "Mistral Large",
    "Command-Xlarge-Beta",
]
FAMILY_ORDER = ["Claude", "GPT", "Llama 3", "Mistral", "Command"]
DIMENSION_ORDER = [
    "Conciseness",
    "Completeness",
    "Faithfulness",
    "Helpfulness",
    "Logical robustness",
    "Logical correctness",
    "Understandability",
    "Harmlessness",
]

# z for ~90% central interval (R used qnorm(0.95) for +/-)
Z_90 = 1.6448536269514722

# Dimensions expected as CSV filenames inside data/
DEFAULT_DIMENSIONS = [
    "Completeness",
    "Logical correctness",
    "Helpfulness",
    "Logical robustness",
    "Faithfulness",
    "Conciseness",
]