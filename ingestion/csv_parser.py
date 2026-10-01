"""
CSV data ingestion utilities for the Bitcoin AI Monitoring System.

This module:
- Loads transaction CSV files
- Loads network-event CSV files
- Validates required columns
- Parses timestamps
- Converts scalar numeric fields
- Detects malformed required values
- Returns pandas DataFrames

The module intentionally does not perform heavy preprocessing.
Cleaning, normalization and feature engineering belong to later stages.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Expected schemas
# -------------------------------------------------------------------

TRANSACTION_COLUMNS = [
    "timestamp",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
]

NETWORK_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "txid",
    "geo_country",
    "asn",
]


# -------------------------------------------------------------------
# Generic CSV loader
# -------------------------------------------------------------------

def load_csv(
    file_path: str | Path,
    required_columns: list[str],
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and validate a CSV file.

    Parameters
    ----------
    file_path:
        Path to the CSV file.

    required_columns:
        Columns that must exist in the CSV.

    numeric_columns:
        Columns that must contain numeric values.

    Returns
    -------
    pandas.DataFrame
        Validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.

    ValueError
        If the file is empty, required columns are missing,
        timestamps are invalid, or numeric values are malformed.
    """

    file_path = Path(file_path)

    # Check that the supplied path exists.
    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    # Prevent accidentally passing a directory.
    if not file_path.is_file():
        raise ValueError(
            f"Expected a file, but received: {file_path}"
        )

    # Load CSV.
    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to read CSV file '{file_path}': {exc}"
        ) from exc

    # Empty file check.
    if df.empty:
        raise ValueError(
            f"CSV file contains no data rows: {file_path}"
        )

    # ---------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------------
    # Validate transaction/network identifier
    # ---------------------------------------------------------------

    if "txid" in df.columns:
        df["txid"] = df["txid"].astype("string").str.strip()

        invalid_txids = df["txid"].isna() | (df["txid"] == "")

        if invalid_txids.any():
            invalid_count = int(invalid_txids.sum())

            raise ValueError(
                f"Found {invalid_count} row(s) with missing or empty txid."
            )

    # ---------------------------------------------------------------
    # Parse timestamp
    # ---------------------------------------------------------------

    if "timestamp" in df.columns:
        original_timestamp = df["timestamp"].copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

        invalid_timestamps = (
            original_timestamp.notna()
            & df["timestamp"].isna()
        )

        if invalid_timestamps.any():
            invalid_count = int(invalid_timestamps.sum())

            raise ValueError(
                f"Found {invalid_count} invalid timestamp value(s)."
            )

    # ---------------------------------------------------------------
    # Convert numeric columns
    # ---------------------------------------------------------------

    numeric_columns = numeric_columns or []

    for column in numeric_columns:

        if column not in df.columns:
            continue

        original_values = df[column].copy()

        converted = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        # A value is malformed when it was present but could not
        # be converted to a number.
        invalid_values = (
            original_values.notna()
            & converted.isna()
        )

        if invalid_values.any():
            invalid_count = int(invalid_values.sum())

            raise ValueError(
                f"Column '{column}' contains "
                f"{invalid_count} malformed numeric value(s)."
            )

        df[column] = converted

    return df


# -------------------------------------------------------------------
# Transaction loader
# -------------------------------------------------------------------

def load_transactions(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load a Bitcoin transaction CSV.

    Scalar numeric fields such as 'fee' are converted to numeric.
    Address and amount-list fields remain in their original textual
    representation because their structured parsing belongs to the
    preprocessing stage.
    """

    return load_csv(
        file_path=file_path,
        required_columns=TRANSACTION_COLUMNS,
        numeric_columns=["fee"],
    )


# -------------------------------------------------------------------
# Network-event loader
# -------------------------------------------------------------------

def load_network_events(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load a Bitcoin network-event CSV.

    Source/destination ports and ASN are converted to numeric values.
    """

    return load_csv(
        file_path=file_path,
        required_columns=NETWORK_COLUMNS,
        numeric_columns=[
            "src_port",
            "dst_port",
            "asn",
        ],
    )


# -------------------------------------------------------------------
# Simple command-line test
# -------------------------------------------------------------------

if __name__ == "__main__":

    transaction_file = (
        "data/raw/synthetic_transactions.csv"
    )

    network_file = (
        "data/raw/synthetic_network_events.csv"
    )

    print("=" * 60)
    print("BITCOIN AI MONITORING - DATA INGESTION TEST")
    print("=" * 60)

    transactions = load_transactions(transaction_file)
    network_events = load_network_events(network_file)

    print("\nTransaction dataset:")
    print(f"Rows: {len(transactions)}")
    print(f"Columns: {len(transactions.columns)}")

    print("\nNetwork dataset:")
    print(f"Rows: {len(network_events)}")
    print(f"Columns: {len(network_events.columns)}")

    common_txids = set(transactions["txid"]).intersection(
        set(network_events["txid"])
    )

    print("\nCorrelation key check:")
    print(f"Common txids: {len(common_txids)}")

    print("\nTransaction dtypes:")
    print(transactions.dtypes)

    print("\nNetwork dtypes:")
    print(network_events.dtypes)

    print("\nData ingestion completed successfully.")
