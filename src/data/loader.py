"""
Data loading module for the rent prediction project.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    MIN_PRICE,
    MIN_FLOOR_SPACE,
    MAX_FLOOR_SPACE,
)

logger = logging.getLogger(__name__)


class DataLoadError(Exception):

    pass


class SchemaValidationError(Exception):

    pass


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load raw apartment data from CSV file.

    Args:
        path: Path to CSV file. If None, uses default path from config.

    Returns:
        DataFrame with raw apartment data.
    """

    file_path = Path(path) if path else RAW_DATA_PATH

    logger.info(f"Loading data from {file_path}")

    if not file_path.exists():
        raise DataLoadError(f"Data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise DataLoadError(f"Failed to read CSV file: {e}") from e

    logger.info(f"Loaded {len(df)} records")

    # Validate schema
    _validate_schema(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has all required columns.

    Args:
        df: DataFrame to validate.
    """

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise SchemaValidationError(
            f"Missing required columns: {missing_columns}"
        )

    logger.debug("Schema validation passed")


def validate_data_quality(df: pd.DataFrame) -> dict[str, any]:
    """
    Check data quality and return statistics.

    Args:
        df: DataFrame to check.

    Returns:
        Dictionary with quality metrics.
    """

    stats = {
        "total_records": len(df),
        "missing_values_total": df.isnull().sum().sum(),
        "missing_values_by_column": df.isnull().sum().to_dict(),
        "duplicate_ids": df["id"].duplicated().sum(),
        "price_stats": {
            "min": df[TARGET_COLUMN].min(),
            "max": df[TARGET_COLUMN].max(),
            "mean": df[TARGET_COLUMN].mean(),
            "median": df[TARGET_COLUMN].median(),
        },
        "invalid_prices": len(df[df[TARGET_COLUMN] < MIN_PRICE]),
        "invalid_floor_space": len(
            df[
                (df["floor_space"] < MIN_FLOOR_SPACE)
                | (df["floor_space"] > MAX_FLOOR_SPACE)
            ]
        ),
    }

    logger.info(f"Data quality check: {stats['total_records']} records, "
                f"{stats['missing_values_total']} missing values")

    return stats


def load_and_shuffle(
    path: Path | str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load data and shuffle for training.

    Args:
        path: Path to CSV file. If None, uses default.
        random_state: Random seed for shuffling.

    Returns:
        Shuffled DataFrame.
    """

    df = load_raw_data(path)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Data shuffled with random_state={random_state}")

    return df_shuffled


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Args:
        df: Input DataFrame.
        target_column: Name of target column.

    Returns:
        Tuple of (features DataFrame, target Series).
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.debug(f"Split data: X shape {X.shape}, y shape {y.shape}")

    return X, y


# Helper functions

def load_data_for_training(
    path: Path | str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare data for training in one step.

    Args:
        path: Path to CSV file. If None, uses default.
        random_state: Random seed for shuffling.

    Returns:
        Tuple of (features DataFrame, target Series).

    """
    df = load_and_shuffle(path, random_state)
    return get_feature_target_split(df)


if __name__ == "__main__":
    # Quick test when running directly
    logging.basicConfig(level=logging.INFO)

    print("Testing data loader...")

    df = load_raw_data()
    print(f"\nLoaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")

    stats = validate_data_quality(df)
    print(f"\nData quality stats:")
    print(f"  - Missing values: {stats['missing_values_total']}")
    print(f"  - Invalid prices: {stats['invalid_prices']}")
    print(f"  - Price range: {stats['price_stats']['min']:.0f} - {stats['price_stats']['max']:.0f} CZK")

    X, y = load_data_for_training()
    print(f"\nReady for training:")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")
