"""
Evaluation metrics module.

This module provides functions for calculating and logging model metrics.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""

    mae: float
    rmse: float
    r2: float
    # Mean Absolute Percentage Error
    mape: float
    # Median Absolute Error
    median_ae: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "median_ae": self.median_ae,
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"MAE: {self.mae:.2f} Kč | "
            f"RMSE: {self.rmse:.2f} Kč | "
            f"R²: {self.r2:.4f} | "
            f"MAPE: {self.mape:.2f}%"
        )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Calculate all regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RegressionMetrics with all calculated metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    median_ae = np.median(np.abs(y_true - y_pred))

    metrics = RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        median_ae=median_ae,
    )

    logger.info(f"Metrics: {metrics}")

    return metrics


def compare_models(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, RegressionMetrics]:
    """
    Compare multiple models.

    Args:
        results: Dict mapping model name to (y_true, y_pred) tuple.

    Returns:
        Dict mapping model name to RegressionMetrics.

    """
    comparison = {}

    for name, (y_true, y_pred) in results.items():
        comparison[name] = calculate_metrics(y_true, y_pred)

    # Log comparison
    logger.info("Model comparison:")
    for name, metrics in comparison.items():
        logger.info(f"  {name}: {metrics}")

    return comparison


def calculate_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculate error distribution statistics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary with error distribution stats.
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    return {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "percentile_25": float(np.percentile(abs_errors, 25)),
        "percentile_50": float(np.percentile(abs_errors, 50)),
        "percentile_75": float(np.percentile(abs_errors, 75)),
        "percentile_90": float(np.percentile(abs_errors, 90)),
        "percentile_95": float(np.percentile(abs_errors, 95)),
        "under_1000": float(np.mean(abs_errors < 1000) * 100),
        "under_2500": float(np.mean(abs_errors < 2500) * 100),
        "under_5000": float(np.mean(abs_errors < 5000) * 100),
    }


def log_metrics_to_mlflow(
    metrics: RegressionMetrics,
    prefix: str = "",
) -> None:
    """
    Log metrics to MLflow.

    Args:
        metrics: Metrics to log.
        prefix: Prefix for metric names (e.g., "train_", "test_").
    """
    try:
        import mlflow

        mlflow.log_metrics({
            f"{prefix}mae": metrics.mae,
            f"{prefix}rmse": metrics.rmse,
            f"{prefix}r2": metrics.r2,
            f"{prefix}mape": metrics.mape,
            f"{prefix}median_ae": metrics.median_ae,
        })

        logger.debug(f"Metrics logged to MLflow with prefix '{prefix}'")

    except ImportError:
        logger.warning("MLflow not installed, skipping metric logging")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
