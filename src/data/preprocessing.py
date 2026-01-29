"""
Data preprocessing module for the rent prediction project.

This module handles all data transformations: cleaning, encoding, and feature engineering.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    COLUMNS_TO_DROP,
    EQUIPMENT_MAPPING,
    MAX_FLOOR_SPACE,
    MAX_PRICE_QUANTILE,
    MIN_FLOOR_SPACE,
    MIN_PRICE,
    ONEHOT_ENCODING_COLUMNS,
    PENB_MAPPING,
    TARGET_COLUMN,
    TARGET_ENCODING_COLUMNS,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline."""

    X: np.ndarray
    y: pd.Series
    feature_names: list[str]
    target_encoder: ce.TargetEncoder
    preprocessor: ColumnTransformer
    n_samples_before: int
    n_samples_after: int


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with sensible defaults.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with filled missing values.
    """
    df = df.copy()

    df["condition"] = df["condition"].fillna("UNDEFINED")
    df["disposition"] = df["disposition"].fillna("UNDEFINED")
    df["equipment"] = df["equipment"].fillna("UNDEFINED")
    df["penb"] = df["penb"].fillna("G")
    df["building_type"] = df["building_type"].fillna("UNDEFINED")

    df["land_space"] = df["land_space"].fillna(0)

    logger.debug("Missing values filled")

    return df


def remove_outliers(
    df: pd.DataFrame,
    price_quantile: float = MAX_PRICE_QUANTILE,
) -> pd.DataFrame:
    """
    Remove outliers based on price and floor space thresholds.

    Args:
        df: Input DataFrame.
        price_quantile: Upper quantile for price (default 0.99).

    Returns:
        DataFrame with outliers removed.
    """
    n_before = len(df)

    max_price = df[TARGET_COLUMN].quantile(price_quantile)

    mask = (
        (df[TARGET_COLUMN] >= MIN_PRICE)
        & (df[TARGET_COLUMN] <= max_price)
        & (df["floor_space"] >= MIN_FLOOR_SPACE)
        & (df["floor_space"] <= MAX_FLOOR_SPACE)
    )

    df_clean = df[mask].copy()
    n_after = len(df_clean)

    logger.info(f"Removed {n_before - n_after} outliers ({(n_before - n_after) / n_before * 100:.1f}%)")

    return df_clean


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not used for modeling.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns dropped.
    """
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    logger.debug(f"Dropped columns: {cols_to_drop}")

    return df


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ordinal encoding to equipment and penb columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with ordinal encoding applied.
    """
    df = df.copy()

    if "equipment" in df.columns:
        df["equipment"] = df["equipment"].map(EQUIPMENT_MAPPING)

    if "penb" in df.columns:
        df["penb"] = df["penb"].map(PENB_MAPPING)

    logger.debug("Ordinal encoding applied")

    return df


def apply_target_encoding(
    df: pd.DataFrame,
    target: pd.Series,
    encoder: ce.TargetEncoder | None = None,
) -> tuple[pd.DataFrame, ce.TargetEncoder]:
    """
    Apply target encoding to high-cardinality categorical columns.

    Args:
        df: Input DataFrame.
        target: Target Series for fitting encoder.
        encoder: Pre-fitted encoder (for inference). If None, fits new encoder.

    Returns:
        Tuple of (encoded DataFrame, fitted encoder).
    """
    df = df.copy()
    cols = [c for c in TARGET_ENCODING_COLUMNS if c in df.columns]

    if encoder is None:
        encoder = ce.TargetEncoder(cols=cols)
        df[cols] = encoder.fit_transform(df[cols], target)
        logger.debug(f"Target encoder fitted on columns: {cols}")
    else:
        df[cols] = encoder.transform(df[cols])
        logger.debug(f"Target encoder applied to columns: {cols}")

    return df, encoder


def build_sklearn_preprocessor(
    X: pd.DataFrame,
    use_scaler: bool = False,
) -> tuple[ColumnTransformer, list[str]]:
    """
    Build sklearn ColumnTransformer for final preprocessing.

    Args:
        X: Features DataFrame (after target encoding).
        use_scaler: Whether to include StandardScaler (for neural networks).

    Returns:
        Tuple of (fitted preprocessor, feature names).
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in ONEHOT_ENCODING_COLUMNS if c in X.columns]

    # Numeric pipeline
    if use_scaler:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
        ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Fit to get feature names
    preprocessor.fit(X)

    # Get feature names
    cat_feature_names = []
    if categorical_features:
        cat_feature_names = list(
            preprocessor.named_transformers_["cat"]["onehot"]
            .get_feature_names_out(categorical_features)
        )

    feature_names = numeric_features + cat_feature_names

    logger.info(f"Preprocessor built: {len(feature_names)} features (scaler={use_scaler})")

    return preprocessor, feature_names


def preprocess_data(
    df: pd.DataFrame,
    use_scaler: bool = False,
    target_encoder: ce.TargetEncoder | None = None,
) -> PreprocessingResult:
    """
    Full preprocessing pipeline for training data.

    Args:
        df: Raw DataFrame.
        use_scaler: Whether to include StandardScaler (True for neural networks).
        target_encoder: Pre-fitted encoder (for inference). If None, fits new.

    Returns:
        PreprocessingResult with processed data and fitted transformers.

    Example:
        >>> from src.data.loader import load_raw_data
        >>> df = load_raw_data()
        >>> result = preprocess_data(df, use_scaler=False)
        >>> print(f"Shape: {result.X.shape}")
    """
    n_samples_before = len(df)

    df = fill_missing_values(df)

    df = remove_outliers(df)

    df = drop_unused_columns(df)

    df = encode_ordinal_features(df)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    X, target_encoder = apply_target_encoding(X, y, target_encoder)

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    preprocessor, feature_names = build_sklearn_preprocessor(X, use_scaler)

    X_processed = preprocessor.transform(X)

    n_samples_after = len(X_processed)

    logger.info(f"Preprocessing complete: {n_samples_before} -> {n_samples_after} samples")

    return PreprocessingResult(
        X=X_processed,
        y=y.reset_index(drop=True),
        feature_names=feature_names,
        target_encoder=target_encoder,
        preprocessor=preprocessor,
        n_samples_before=n_samples_before,
        n_samples_after=n_samples_after,
    )


def preprocess_single_sample(
    data: dict,
    target_encoder: ce.TargetEncoder,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """
    Preprocess a single sample for inference.

    Args:
        data: Dictionary with feature values.
        target_encoder: Fitted target encoder.
        preprocessor: Fitted sklearn preprocessor.

    Returns:
        Numpy array ready for model prediction.
    """
    df = pd.DataFrame([data])

    df = fill_missing_values(df)

    df = encode_ordinal_features(df)

    cols = [c for c in TARGET_ENCODING_COLUMNS if c in df.columns]
    df[cols] = target_encoder.transform(df[cols])

    X = preprocessor.transform(df)

    return X


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.loader import load_raw_data

    print("Testing preprocessing pipeline...")

    df = load_raw_data()
    print(f"\nRaw data: {len(df)} records")

    # Test XGBoost preprocessing
    result = preprocess_data(df, use_scaler=False)
    print(f"\nXGBoost preprocessing (no scaler):")
    print(f"  - Shape: {result.X.shape}")
    print(f"  - Features: {len(result.feature_names)}")
    print(f"  - Samples: {result.n_samples_before} -> {result.n_samples_after}")

    # Test Neural Network preprocessing
    result_nn = preprocess_data(df, use_scaler=True)
    print(f"\nNeural Network preprocessing (with scaler):")
    print(f"  - Shape: {result_nn.X.shape}")
    print(f"  - First row (scaled): {result_nn.X[0, :3]}")
