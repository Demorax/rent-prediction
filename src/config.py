"""
Configuration module for the rent prediction project.

Contains all constants, paths, and hyperparameters in one place.
"""

from pathlib import Path
from typing import Final

# =============================================================================
# PATHS
# =============================================================================

# Project root directory (2 levels up from this file)
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent

# Data paths
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_PATH: Final[Path] = DATA_DIR / "apartment_properties_rent.csv"

# Model paths
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
XGBOOST_MODEL_PATH: Final[Path] = MODELS_DIR / "xgboost.json"
MLP_MODEL_PATH: Final[Path] = MODELS_DIR / "multilayer_perceptron.keras"
CNN_MODEL_PATH: Final[Path] = MODELS_DIR / "convolutional_neural_network.keras"

# =============================================================================
# DATA SCHEMA
# =============================================================================

# Required columns in raw data
REQUIRED_COLUMNS: Final[list[str]] = [
    "id",
    "building_type",
    "city",
    "condition",
    "estate_type",
    "floor_space",
    "land_space",
    "price",
    "region",
    "sale_type",
    "source",
    "disposition",
    "equipment",
    "penb",
]

# Columns to drop during preprocessing
COLUMNS_TO_DROP: Final[list[str]] = ["estate_type", "sale_type", "source", "id"]

# Target column
TARGET_COLUMN: Final[str] = "price"

# =============================================================================
# CATEGORICAL MAPPINGS
# =============================================================================

# Equipment levels (ordinal encoding)
EQUIPMENT_MAPPING: Final[dict[str, int]] = {
    "UNDEFINED": 0,
    "UNFURNISHED": 1,
    "PARTIALLY": 2,
    "FURNISHED": 3,
}

# Energy performance certificate (ordinal encoding)
PENB_MAPPING: Final[dict[str, int]] = {
    "G": 0,  # Worst
    "F": 1,
    "E": 2,
    "D": 3,
    "C": 4,
    "B": 5,
    "A": 6,  # Best
}

# Valid building types
VALID_BUILDING_TYPES: Final[list[str]] = [
    "BRICK",
    "PANEL",
    "WOODEN",
    "LOW_ENERGY",
    "UNDEFINED",
]

# Valid conditions
VALID_CONDITIONS: Final[list[str]] = [
    "NEW",
    "VERY_GOOD",
    "GOOD",
    "BAD",
    "UNDER_CONSTRUCTION",
    "BEFORE_RECONSTRUCTION",
    "AFTER_RECONSTRUCTION",
    "UNDEFINED",
]

# Columns for target encoding (high cardinality)
TARGET_ENCODING_COLUMNS: Final[list[str]] = ["city", "region", "disposition"]

# Columns for one-hot encoding (low cardinality)
ONEHOT_ENCODING_COLUMNS: Final[list[str]] = ["condition", "building_type"]

# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================

# Price thresholds (CZK per month)
MIN_PRICE: Final[int] = 1_000
MAX_PRICE_QUANTILE: Final[float] = 0.99  # Remove top 1% outliers

# Floor space thresholds (m²)
MIN_FLOOR_SPACE: Final[int] = 10
MAX_FLOOR_SPACE: Final[int] = 250

# =============================================================================
# MODEL HYPERPARAMETERS (defaults)
# =============================================================================

XGBOOST_DEFAULT_PARAMS: Final[dict] = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5,
    "random_state": 42,
    "n_jobs": -1,
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 42
CV_FOLDS: Final[int] = 5
