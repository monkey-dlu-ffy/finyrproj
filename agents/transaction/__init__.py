"""Transaction agent: anomaly signals over structured ledger rows.

Owns the gradient-boosted detectors (xgboost / lightgbm) and their SHAP
attributions. Consumes data/processed, writes scored cases + per-feature
contributions.
"""
