"""Build a baseline wait-time lookup table for PortAventura rides.

Reads data/processed/portaventura_wait_times.parquet (built by
build_portaventura_dataset.py) and computes median WAIT_TIME_MAX grouped by
ride, hour of day, day of week, and month. Saves the result to
data/processed/baseline_wait_lookup.parquet.
"""

import pandas as pd

IN_PATH = "data/processed/portaventura_wait_times.parquet"
OUT_PATH = "data/processed/baseline_wait_lookup.parquet"


def main() -> None:
    df = pd.read_parquet(IN_PATH)

    df["day_of_week"] = df["WORK_DATE"].dt.dayofweek  # 0=Monday .. 6=Sunday
    df["month"] = df["WORK_DATE"].dt.month

    lookup = (
        df.groupby(
            ["ENTITY_DESCRIPTION_SHORT", "DEB_TIME_HOUR", "day_of_week", "month"]
        )["WAIT_TIME_MAX"]
        .agg(median_wait="median", n="size")
        .reset_index()
        .rename(columns={"ENTITY_DESCRIPTION_SHORT": "ride", "DEB_TIME_HOUR": "hour"})
    )

    lookup.to_parquet(OUT_PATH, index=False)

    print(f"Baseline lookup rows: {len(lookup):,}")
    print(f"Rides: {lookup['ride'].nunique()}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
