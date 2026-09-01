"""Phase 1 data foundation: feature engineering, typology tagging, splits.

Every function that fits a statistic (category amount mean/std, typology
percentile thresholds) takes fraudTrain only -- callers must not fit on test.
"""
