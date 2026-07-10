"""Central configuration and filesystem paths for Alchemind."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
WEB_DIR = PACKAGE_ROOT / "web"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default reference set used for novelty checks (compounds considered "known").
REFERENCE_SMILES_FILE = DATA_DIR / "reference_smiles.txt"

# Bundled sample solubility dataset (SMILES, measured log-solubility).
SOLUBILITY_DATASET = DATA_DIR / "solubility_sample.csv"
SOLUBILITY_MODEL = MODELS_DIR / "solubility_rf.joblib"

# Generation defaults
DEFAULT_POPULATION = 60
DEFAULT_GENERATIONS = 8
DEFAULT_MUTATION_RATE = 0.5
RANDOM_SEED = 42
