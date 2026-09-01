"""Train/val split. fraudTest is never touched here -- it stays fully held out
until final evaluation, matching Sparkov's own time-based train/test cut."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED


def train_val_split(
    df: pd.DataFrame, val_size: float, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(
        df, test_size=val_size, stratify=df["is_fraud"], random_state=seed
    )
    return train, val
