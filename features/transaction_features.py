"""
Transaction feature engineering for the Bitcoin AI Monitoring System.

This module creates explainable transaction-level features from the
cleaned Bitcoin transaction dataset.

The generated features are intended for:
- anomaly detection
- risk scoring
- investigation explanations
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = "data/processed/correlated_data.csv"
OUTPUT_FILE = "data/processed/transaction_features.csv"


def split_values(value):
    """
    Convert a possibly separator-based field into a list of values.

    The current synthetic dataset may contain a single value, while
    future datasets may contain multiple comma/semicolon/pipe-separated
    values.
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    for separator in ["|", ";"]:
        value = value.replace(separator, ",")

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def count_values(value):
    """Return the number of values in a field."""

    return len(split_values(value))


def create_transaction_features(df):
    """
    Create transaction-level features.

    Parameters
    ----------
    df:
        Correlated transaction/network DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing transaction features.
    """

    required_columns = [
        "txid",
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts",
        "fee",
        "transaction_timestamp",
    ]

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

    features = pd.DataFrame(index=df.index)

    # ---------------------------------------------------------------
    # Basic identifiers
    # ---------------------------------------------------------------

    features["txid"] = df["txid"].astype("string").str.strip()

    # ---------------------------------------------------------------
    # Amount features
    # ---------------------------------------------------------------

    features["input_amount"] = pd.to_numeric(
        df["input_amounts"],
        errors="coerce",
    )

    features["output_amount"] = pd.to_numeric(
        df["output_amounts"],
        errors="coerce",
    )

    features["fee"] = pd.to_numeric(
        df["fee"],
        errors="coerce",
    )

    features["amount_difference"] = (
        features["input_amount"]
        - features["output_amount"]
    )

    features["input_output_ratio"] = (
        features["output_amount"]
        / features["input_amount"].replace(0, pd.NA)
    )

    features["fee_ratio"] = (
        features["fee"]
        / features["input_amount"].replace(0, pd.NA)
    )

    # ---------------------------------------------------------------
    # Address count features
    # ---------------------------------------------------------------

    features["input_address_count"] = (
        df["input_addresses"].apply(count_values)
    )

    features["output_address_count"] = (
        df["output_addresses"].apply(count_values)
    )

    # ---------------------------------------------------------------
    # Transaction timing
    # ---------------------------------------------------------------

    timestamp = pd.to_datetime(
        df["transaction_timestamp"],
        errors="coerce",
        utc=True,
    )

    features["transaction_hour"] = timestamp.dt.hour

    features["transaction_day_of_week"] = (
        timestamp.dt.dayofweek
    )

    features["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    # ---------------------------------------------------------------
    # Address activity / reuse
    # ---------------------------------------------------------------

    input_addresses = df[
        "input_addresses"
    ].apply(split_values)

    output_addresses = df[
        "output_addresses"
    ].apply(split_values)

    input_address_counts = {}

    output_address_counts = {}

    for addresses in input_addresses:
        for address in addresses:
            input_address_counts[address] = (
                input_address_counts.get(address, 0) + 1
            )

    for addresses in output_addresses:
        for address in addresses:
            output_address_counts[address] = (
                output_address_counts.get(address, 0) + 1
            )

    features["input_address_reuse_count"] = (
        input_addresses.apply(
            lambda addresses: max(
                [input_address_counts[a] for a in addresses],
                default=0,
            )
        )
    )

    features["output_address_reuse_count"] = (
        output_addresses.apply(
            lambda addresses: max(
                [output_address_counts[a] for a in addresses],
                default=0,
            )
        )
    )

    # ---------------------------------------------------------------
    # Transaction frequency
    # ---------------------------------------------------------------

    features["input_address_transaction_frequency"] = (
        input_addresses.apply(
            lambda addresses: sum(
                input_address_counts[a]
                for a in addresses
            )
        )
    )

    features["output_address_transaction_frequency"] = (
        output_addresses.apply(
            lambda addresses: sum(
                output_address_counts[a]
                for a in addresses
            )
        )
    )

    # ---------------------------------------------------------------
    # Clean infinite values
    # ---------------------------------------------------------------

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    return features


def main():
    """Run transaction feature engineering."""

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    feature_df = create_transaction_features(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 60)
    print("TRANSACTION FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Input rows     : {len(df)}")
    print(f"Output rows    : {len(feature_df)}")
    print(f"Feature count  : {len(feature_df.columns)}")

    print("\nGenerated features:")
    for column in feature_df.columns:
        print(f"✓ {column}")

    print(
        f"\nSaved to: {output_path}"
    )

    print(
        "\nTransaction feature engineering completed successfully."
    )


if __name__ == "__main__":
    main()
