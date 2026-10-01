"""
Transaction preprocessing for the Bitcoin AI Monitoring System.

This module:
- Loads transaction data using the ingestion layer
- Removes unusable transaction rows
- Normalizes txids and addresses
- Validates timestamps
- Validates amount and fee values
- Removes duplicate transactions
- Saves cleaned data to data/processed/

The goal is clean, consistent data for later pipeline stages.
"""

from pathlib import Path

import pandas as pd

from ingestion.csv_parser import load_transactions


# -------------------------------------------------------------------
# Required transaction columns
# -------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normalize_text(value):
    """
    Normalize a text value by:
    - converting it to string
    - removing leading/trailing whitespace
    """

    if pd.isna(value):
        return value

    return str(value).strip()


def normalize_address_field(value):
    """
    Normalize an address field.

    For the current synthetic dataset, address values may be stored
    either as a single value or as a separator-based list.

    We preserve the original structure while cleaning whitespace.
    """

    if pd.isna(value):
        return value

    value = str(value).strip()

    # Handle common separators without changing the data model.
    parts = []

    for item in value.replace("|", ",").replace(";", ",").split(","):
        item = item.strip()

        if item:
            parts.append(item)

    if not parts:
        return None

    return ",".join(parts)


def validate_non_negative_column(df, column):
    """
    Ensure a numeric column contains no negative values.
    """

    invalid = df[column].notna() & (df[column] < 0)

    if invalid.any():
        count = int(invalid.sum())

        raise ValueError(
            f"Column '{column}' contains {count} negative value(s)."
        )


# -------------------------------------------------------------------
# Main cleaning function
# -------------------------------------------------------------------

def clean_transactions(
    input_path,
    output_path=None,
):
    """
    Load and clean transaction data.

    Parameters
    ----------
    input_path:
        Path to the raw transaction CSV.

    output_path:
        Optional path for saving cleaned transactions.

    Returns
    -------
    pandas.DataFrame
        Cleaned transaction DataFrame.
    """

    input_path = Path(input_path)

    # ---------------------------------------------------------------
    # Load through the ingestion layer
    # ---------------------------------------------------------------

    df = load_transactions(input_path)

    original_rows = len(df)

    # ---------------------------------------------------------------
    # Check schema
    # ---------------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Transaction dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------------
    # Normalize text columns
    # ---------------------------------------------------------------

    text_columns = [
        "txid",
        "input_addresses",
        "output_addresses",
        "script_type",
    ]

    for column in text_columns:
        df[column] = df[column].apply(normalize_text)

    # ---------------------------------------------------------------
    # Normalize address fields
    # ---------------------------------------------------------------

    df["input_addresses"] = df["input_addresses"].apply(
        normalize_address_field
    )

    df["output_addresses"] = df["output_addresses"].apply(
        normalize_address_field
    )

    # ---------------------------------------------------------------
    # Normalize timestamp
    # ---------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------------
    # Validate numeric fields
    # ---------------------------------------------------------------

    numeric_columns = [
        "input_amounts",
        "output_amounts",
        "fee",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # Remove rows missing critical identifiers
    # ---------------------------------------------------------------

    critical_columns = [
        "timestamp",
        "txid",
    ]

    before_required_cleanup = len(df)

    df = df.dropna(
        subset=critical_columns
    ).copy()

    removed_required = (
        before_required_cleanup - len(df)
    )

    # Empty txid check after normalization.
    empty_txid = df["txid"].eq("")

    if empty_txid.any():
        df = df.loc[~empty_txid].copy()

    # ---------------------------------------------------------------
    # Remove duplicate transaction IDs
    # ---------------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["txid"],
        keep="first",
    ).copy()

    duplicate_rows_removed = (
        before_duplicates - len(df)
    )

    # ---------------------------------------------------------------
    # Validate numeric values
    # ---------------------------------------------------------------

    # Negative fees are invalid for this project.
    validate_non_negative_column(df, "fee")

    # Negative transaction amounts are invalid.
    validate_non_negative_column(
        df,
        "input_amounts",
    )

    validate_non_negative_column(
        df,
        "output_amounts",
    )

    # ---------------------------------------------------------------
    # Sort by timestamp
    # ---------------------------------------------------------------

    df = df.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save cleaned dataset
    # ---------------------------------------------------------------

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
        )

    # ---------------------------------------------------------------
    # Processing summary
    # ---------------------------------------------------------------

    cleaned_rows = len(df)

    print("=" * 60)
    print("TRANSACTION PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Input rows           : {original_rows}")
    print(f"Output rows          : {cleaned_rows}")
    print(f"Rows removed         : {original_rows - cleaned_rows}")
    print(f"Missing-ID rows      : {removed_required}")
    print(f"Duplicate TXIDs removed: {duplicate_rows_removed}")

    print("\nValidation:")
    print("✓ Required columns validated")
    print("✓ TXIDs normalized")
    print("✓ Addresses normalized")
    print("✓ Timestamps normalized")
    print("✓ Numeric fields validated")
    print("✓ Duplicate TXIDs removed")
    print("✓ Transactions sorted by timestamp")

    return df


# -------------------------------------------------------------------
# Command-line test
# -------------------------------------------------------------------

if __name__ == "__main__":

    input_file = (
        "data/raw/synthetic_transactions.csv"
    )

    output_file = (
        "data/processed/clean_transactions.csv"
    )

    cleaned = clean_transactions(
        input_path=input_file,
        output_path=output_file,
    )

    print("\nCleaned dataset:")
    print(f"Rows    : {len(cleaned)}")
    print(f"Columns : {len(cleaned.columns)}")

    print("\nFirst 5 rows:")
    print(cleaned.head())

    print(
        f"\nSaved to: {output_file}"
    )

    print(
        "\nTransaction preprocessing completed successfully."
    )
