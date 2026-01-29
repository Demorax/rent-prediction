"""
Training pipeline script.

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --optimize --trials 50
"""

import argparse
import logging

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_data
from src.models.train import optimize_model, save_artifacts, save_model, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(optimize: bool = False, n_trials: int = 100):
    logger.info("Loading data...")
    df = load_raw_data()

    logger.info("Preprocessing...")
    result = preprocess_data(df, use_scaler=False)

    X_train, X_test, y_train, y_test = train_test_split(
        result.X, result.y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    params = None
    if optimize:
        logger.info(f"Running Optuna optimization ({n_trials} trials)...")
        params = optimize_model(X_train, y_train, n_trials=n_trials)
        logger.info(f"Best params: {params}")

    logger.info("Training model...")
    training_result = train_model(X_train, y_train, X_test, y_test, params=params)

    logger.info(f"Results: MAE={training_result.test_mae:.2f}, R2={training_result.test_r2:.4f}")

    save_model(training_result.model)
    save_artifacts(result.preprocessor, result.target_encoder, result.feature_names)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args()

    main(optimize=args.optimize, n_trials=args.trials)
