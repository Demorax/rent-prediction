import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, XGBOOST_DEFAULT_PARAMS, XGBOOST_MODEL_PATH

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: xgb.XGBRegressor
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    best_params: dict
    n_iterations: int


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
    use_gpu: bool = True,
    early_stopping_rounds: int = 50,
) -> TrainingResult:
    model_params = XGBOOST_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)

    if use_gpu:
        model_params["device"] = "cuda"
        model_params["tree_method"] = "hist"

    model = xgb.XGBRegressor(
        **model_params,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="mae"
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return TrainingResult(
        model=model,
        train_mae=mean_absolute_error(y_train, y_pred_train),
        test_mae=mean_absolute_error(y_test, y_pred_test),
        train_r2=r2_score(y_train, y_pred_train),
        test_r2=r2_score(y_test, y_pred_test),
        best_params=model_params,
        n_iterations=model.best_iteration if hasattr(model, 'best_iteration') else 0,
    )


def optimize_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 100,
    use_gpu: bool = True,
    cv_folds: int = CV_FOLDS,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "n_estimators": 1000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        if use_gpu:
            params["device"] = "cuda"
            params["tree_method"] = "hist"

        model = xgb.XGBRegressor(**params)
        kfold = KFold(n_splits=cv_folds, random_state=RANDOM_STATE, shuffle=True)
        scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error", n_jobs=1)
        return -np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def save_model(model: xgb.XGBRegressor, path: Path | str | None = None) -> Path:
    save_path = Path(path) if path else XGBOOST_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_path))
    logger.info(f"Model saved: {save_path}")
    return save_path


def save_artifacts(preprocessor, target_encoder, feature_names: list[str], path: Path | str | None = None) -> Path:
    save_path = Path(path) if path else MODELS_DIR / "preprocessor.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "feature_names": feature_names,
    }
    joblib.dump(artifacts, save_path)
    logger.info(f"Artifacts saved: {save_path}")
    return save_path


def load_model(path: Path | str | None = None) -> xgb.XGBRegressor:
    load_path = Path(path) if path else XGBOOST_MODEL_PATH
    if not load_path.exists():
        raise FileNotFoundError(f"Model not found: {load_path}")

    model = xgb.XGBRegressor()
    model.load_model(str(load_path))
    return model


def load_artifacts(path: Path | str | None = None) -> dict:
    load_path = Path(path) if path else MODELS_DIR / "preprocessor.joblib"
    if not load_path.exists():
        raise FileNotFoundError(f"Artifacts not found: {load_path}")
    return joblib.load(load_path)
