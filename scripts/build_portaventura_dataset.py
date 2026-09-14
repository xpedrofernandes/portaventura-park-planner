"""Build a filtered, PortAventura-only waiting_times dataset as parquet.

Reads data/raw/waiting_times.csv in chunks (it's ~365MB) with only the
columns needed, keeps PortAventura World rides, drops rows where the ride
was closed (OPEN_TIME == 0 and CAPACITY == 0), and writes the result to
data/processed/portaventura_wait_times.parquet.
"""

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
OUT_PATH = f"{OUT_DIR}/portaventura_wait_times.parquet"

USECOLS = [
    "WORK_DATE",
    "DEB_TIME_HOUR",
    "ENTITY_DESCRIPTION_SHORT",
    "WAIT_TIME_MAX",
    "NB_UNITS",
    "OPEN_TIME",
    "CAPACITY",
]
CHUNKSIZE = 1_000_000


def main() -> None:
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    link = pd.read_csv(f"{RAW_DIR}/link_attraction_park.csv", sep=";")
    pav_rides = set(link.loc[link["PARK"] == "PortAventura World", "ATTRACTION"])

    kept_chunks = []
    total_rows_read = 0
    for chunk in pd.read_csv(
        f"{RAW_DIR}/waiting_times.csv", usecols=USECOLS, chunksize=CHUNKSIZE
    ):
        total_rows_read += len(chunk)
        chunk = chunk[chunk["ENTITY_DESCRIPTION_SHORT"].isin(pav_rides)]
        closed = (chunk["OPEN_TIME"] == 0) & (chunk["CAPACITY"] == 0)
        chunk = chunk[~closed]
        kept_chunks.append(chunk)

    df = pd.concat(kept_chunks, ignore_index=True)
    df["WORK_DATE"] = pd.to_datetime(df["WORK_DATE"])

    df.to_parquet(OUT_PATH, index=False)

    print(f"Rows read from waiting_times.csv: {total_rows_read:,}")
    print(f"Rows kept (PortAventura, open only): {len(df):,}")
    print(f"Date range: {df['WORK_DATE'].min().date()} to {df['WORK_DATE'].max().date()}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
