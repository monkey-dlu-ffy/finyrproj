"""Interactive SQL over the linked ledger Parquet files.

    py -3.13 -i ledger_sql.py          # REPL: q("select ...")
    py -3.13 ledger_sql.py "select ..." # one-shot

Views registered (no data copied -- DuckDB reads the Parquet in place):
    ledger   all 533,009 rows, with row_id and split
    train / val / test   the individual splits
    vendors  the 300-vendor master
Each split's row_id is its positional index, matching the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# DuckDB draws box-art tables; the Windows console defaults to cp1252 and
# raises UnicodeEncodeError on them. Force UTF-8 before any output.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINKED = Path(__file__).resolve().parent / "data" / "linked"

con = duckdb.connect()
for split in ("train", "val", "test"):
    # file_row_number, not row_number() OVER (): an unordered window doesn't
    # guarantee file order, and row_id must be the positional index.
    con.execute(
        f"""CREATE VIEW {split} AS
            SELECT file_row_number AS row_id, * EXCLUDE (file_row_number)
            FROM read_parquet('{(LINKED / f"{split}.parquet").as_posix()}', file_row_number = true)"""
    )
# row_id repeats across splits, so `ledger` carries the split to disambiguate.
con.execute(
    """CREATE VIEW ledger AS
        SELECT *, 'train' AS split FROM train
        UNION ALL SELECT *, 'val' AS split FROM val
        UNION ALL SELECT *, 'test' AS split FROM test"""
)
con.execute(
    f"""CREATE VIEW vendors AS
        SELECT * FROM read_json_auto('{(LINKED / "vendor_master.json").as_posix()}')"""
)


def q(sql: str, limit: int = 50):
    """Run SQL and print it as a table."""
    con.sql(sql).show(max_rows=limit)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q(" ".join(sys.argv[1:]))
    else:
        print(__doc__)
        print("views: ledger, train, val, test, vendors")
        print('try:   q("select * from test where row_id = 45151")')
