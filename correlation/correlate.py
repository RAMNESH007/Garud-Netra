"""
Blockchain and network-event correlation.

This module joins cleaned Bitcoin transaction records with
cleaned network observations using txid as the primary key.

Additional correlation information such as timestamp difference
is calculated for later feature engineering.
"""

from pathlib import Path

import pandas as pd


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


def load_processed_data(
    transaction_path,
    network_path,
):
    """
    Load the two cleaned datasets used for correlation.
    """

    transaction_path = Path(transaction_path)
    network_path = Path(network_path)

    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"Transaction file not found: {transaction_path}"
        )

    if not network_path.is_file():
        raise FileNotFoundError(
            f"Network file not found: {network_path}"
        )

    transactions = pd.read_csv(
        transaction_path,
        parse_dates=["timestamp"],
    )

    network_events = pd.read_csv(
        network_path,
        parse_dates=["timestamp"],
    )

    return transactions, network_events


def validate_columns(df, required_columns, dataset_name):
    """
    Validate that all expected columns exist.
    """

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing)
        )


def correlate_transactions_and_network(
    transaction_path,
    network_path,
    output_path=None,
):
    """
    Correlate transactions with network observations using txid.

    Returns
    -------
    pandas.DataFrame
        Correlated dataset.
    """

    transactions, network_events = load_processed_data(
        transaction_path,
        network_path,
    )

    # ---------------------------------------------------------------
    # Validate schemas
    # ---------------------------------------------------------------

    validate_columns(
        transactions,
        TRANSACTION_COLUMNS,
        "Transaction dataset",
    )

    validate_columns(
        network_events,
        NETWORK_COLUMNS,
        "Network dataset",
    )

    # ---------------------------------------------------------------
    # Normalize txid for safe joining
    # ---------------------------------------------------------------

    transactions["txid"] = (
        transactions["txid"]
        .astype("string")
        .str.strip()
    )

    network_events["txid"] = (
        network_events["txid"]
        .astype("string")
        .str.strip()
    )

    # ---------------------------------------------------------------
    # Make sure timestamps are UTC-aware
    # ---------------------------------------------------------------

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce",
        utc=True,
    )

    network_events["timestamp"] = pd.to_datetime(
        network_events["timestamp"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------------
    # Rename timestamps before merging
    # ---------------------------------------------------------------

    transactions = transactions.rename(
        columns={
            "timestamp": "transaction_timestamp"
        }
    )

    network_events = network_events.rename(
        columns={
            "timestamp": "network_timestamp"
        }
    )

    # ---------------------------------------------------------------
    # Perform txid-based correlation
    # ---------------------------------------------------------------

    correlated = transactions.merge(
        network_events,
        on="txid",
        how="inner",
        suffixes=("_transaction", "_network"),
    )

    # ---------------------------------------------------------------
    # Calculate timing relationship
    # ---------------------------------------------------------------

    correlated["time_difference_seconds"] = (
        correlated["network_timestamp"]
        - correlated["transaction_timestamp"]
    ).dt.total_seconds()

    correlated["absolute_time_difference_seconds"] = (
        correlated["time_difference_seconds"]
        .abs()
    )

    # ---------------------------------------------------------------
    # Sort by transaction time
    # ---------------------------------------------------------------

    correlated = correlated.sort_values(
        by="transaction_timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save result
    # ---------------------------------------------------------------

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        correlated.to_csv(
            output_path,
            index=False,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("=" * 60)
    print("TRANSACTION + NETWORK CORRELATION SUMMARY")
    print("=" * 60)

    print(
        f"Transactions loaded      : {len(transactions)}"
    )

    print(
        f"Network events loaded    : {len(network_events)}"
    )

    print(
        f"Correlated records       : {len(correlated)}"
    )

    print(
        f"Unique correlated TXIDs  : "
        f"{correlated['txid'].nunique()}"
    )

    if not correlated.empty:
        print(
            f"Average time difference : "
            f"{correlated['time_difference_seconds'].mean():.2f} seconds"
        )

        print(
            f"Minimum time difference : "
            f"{correlated['time_difference_seconds'].min():.2f} seconds"
        )

        print(
            f"Maximum time difference : "
            f"{correlated['time_difference_seconds'].max():.2f} seconds"
        )

    print("\nValidation:")
    print("✓ Transaction schema validated")
    print("✓ Network schema validated")
    print("✓ TXIDs normalized")
    print("✓ Timestamps normalized")
    print("✓ Correlation performed using txid")
    print("✓ Timestamp difference calculated")
    print("✓ Results sorted chronologically")

    return correlated


if __name__ == "__main__":

    transaction_file = (
        "data/processed/clean_transactions.csv"
    )

    network_file = (
        "data/processed/clean_network_events.csv"
    )

    output_file = (
        "data/processed/correlated_data.csv"
    )

    correlated_data = correlate_transactions_and_network(
        transaction_path=transaction_file,
        network_path=network_file,
        output_path=output_file,
    )

    print("\nCorrelated dataset:")
    print(
        f"Rows    : {len(correlated_data)}"
    )

    print(
        f"Columns : {len(correlated_data.columns)}"
    )

    print("\nFirst 5 correlated records:")
    print(correlated_data.head())

    print(
        f"\nSaved to: {output_file}"
    )

    print(
        "\nCorrelation completed successfully."
    )
