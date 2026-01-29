"""
Czech Apartment Rent Prediction Package.

This package provides tools for predicting apartment rental prices in the Czech Republic.

Modules:
    - data: Data loading and preprocessing
    - models: Model training and prediction (TODO)
    - evaluation: Metrics and visualization (TODO)
    - api: REST API endpoints (TODO)

Example:
    >>> from src.data.loader import load_raw_data
    >>> from src.data.preprocessing import preprocess_data
    >>>
    >>> df = load_raw_data()
    >>> result = preprocess_data(df, use_scaler=False)
    >>> print(f"Ready for training: {result.X.shape}")
"""

__version__ = "1.0.0"
__author__ = "Tomas"
