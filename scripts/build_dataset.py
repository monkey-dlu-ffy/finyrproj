"""Phase 1 entry point: build data/processed/{train,val,test}.parquet from the
raw Sparkov CSVs, with engineered features and typology tags.

    python scripts/build_dataset.py

Every stat used as a feature or typology threshold is fit on fraudTrain only,
then applied to val/test -- so no test-set distribution leaks into training.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, RAW_DIR, VAL_SIZE, ensure_dirs, seed_everything
from data_pipeline.features import add_all_features, fit_category_amount_stats
from data_pipeline.splits import train_val_split
from data_pipeline.typology import add_typology_columns, fit_typology_thresholds

DUMMY_DIR = Path(__file__).resolve().parent.parent / "dummy" / "k" / "extracted"

DATE_COLS = ["trans_date_trans_time", "dob"]
CATEGORY_COLS = ["merchant", "category", "gender", "state", "job"]

TYPOLOGY_COLS = (
    "typ_online",
    "typ_pos",
    "typ_geo_anomaly",
    "typ_velocity",
    "typ_high_value",
    "typ_doc_mismatch",
)


def _stage_raw() -> tuple[Path, Path]:
    """Copy source CSVs from the scratch staging dir into data/raw/ (the
    documented, gitignored location) if not already there. Copies rather than
    moves -- dummy/ is left untouched so this step is safely re-runnable."""
    train_dst = RAW_DIR / "fraudTrain.csv"
    test_dst = RAW_DIR / "fraudTest.csv"
    if not train_dst.exists():
        shutil.copy2(DUMMY_DIR / "fraudTrain.csv", train_dst)
    if not test_dst.exists():
        shutil.copy2(DUMMY_DIR / "fraudTest.csv", test_dst)
    return train_dst, test_dst


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=DATE_COLS)
    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")
    df["is_fraud"] = df["is_fraud"].astype("int8")
    return df


def main() -> int:
    seed_everything()
    ensure_dirs()

    t0 = time.time()
    train_path, test_path = _stage_raw()
    print(f"staged raw csvs -> {RAW_DIR} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    train_df = _load(train_path)
    test_df = _load(test_path)
    print(f"loaded train={len(train_df):,} test={len(test_df):,} rows ({time.time() - t0:.1f}s)")

    t0 = time.time()
    amount_stats = fit_category_amount_stats(train_df)
    train_df = add_all_features(train_df, amount_stats)
    typ_thresholds = fit_typology_thresholds(train_df)
    train_df = add_typology_columns(train_df, typ_thresholds)
    print(f"engineered + tagged train ({time.time() - t0:.1f}s)")

    t0 = time.time()
    test_df = add_all_features(test_df, amount_stats)
    test_df = add_typology_columns(test_df, typ_thresholds)
    print(f"engineered + tagged test ({time.time() - t0:.1f}s)")

    train_df, val_df = train_val_split(train_df, val_size=VAL_SIZE)
    print(f"split train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, df in splits.items():
        out = PROCESSED_DIR / f"{name}.parquet"
        df.reset_index(drop=True).to_parquet(out, index=False)
        print(f"wrote {out} ({len(df):,} rows)")

    stats_out = PROCESSED_DIR / "feature_stats.json"
    stats_out.write_text(
        json.dumps(
            {
                "category_amount_stats": {
                    str(k): v for k, v in amount_stats.items()
                },
                "typology_thresholds": typ_thresholds,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {stats_out}")

    print("-" * 60)
    for name, df in splits.items():
        rate = df["is_fraud"].mean()
        print(f"{name}: n={len(df):,} fraud_rate={rate:.4%}")
        fraud_rows = df.loc[df["is_fraud"] == 1]
        for col in TYPOLOGY_COLS:
            prevalence = fraud_rows[col].mean() if len(fraud_rows) else float("nan")
            print(f"  {col:<16} fraud_prevalence={prevalence:.2%} (n={int(fraud_rows[col].sum())})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
