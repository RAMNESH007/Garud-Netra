"""
Combined feature engineering for the Bitcoin AI Monitoring System.

This module combines transaction-level and network-level features
into a single ML-ready feature dataset using txid.
"""

from pathlib import Path

import pandas as pd


TRANSACTION_FEATURES_FILE = (
    "data/processed/transaction_features.csv"
)

NETWORK_FEATURES_FILE = (
    "data/processed/network_features.csv"
)

OUTPUT_FILE = (
    "data/processed/combined_features.csv"
)


def load_feature_data(
    transaction_path,
    network_path,
):
    """Load transaction and network feature datasets."""

    transaction_path = Path(transaction_path)
    network_path = Path(network_path)

    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"Transaction features not found: {transaction_path}"
        )

    if not network_path.is_file():
        raise FileNotFoundError(
            f"Network features not found: {network_path}"
        )

    transaction_features = pd.read_csv(
        transaction_path
    )

    network_features = pd.read_csv(
        network_path
    )

    return transaction_features, network_features


def validate_txid(df, dataset_name):
    """Ensure the feature dataset contains valid txids."""

    if "txid" not in df.columns:
        raise ValueError(
            f"{dataset_name} does not contain a txid column."
        )

    if df["txid"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing txid values."
        )

    if df["txid"].astype(str).str.strip().eq("").any():
        raise ValueError(
            f"{dataset_name} contains empty txid values."
        )


def combine_features(
    transaction_path,
    network_path,
    output_path=None,
):
    """
    Combine transaction and network features using txid.

    Returns
    -------
    pandas.DataFrame
        Combined feature dataset.
    """

    transaction_features, network_features = load_feature_data(
        transaction_path,
        network_path,
    )

    # ---------------------------------------------------------------
    # Validate TXIDs
    # ---------------------------------------------------------------

    validate_txid(
        transaction_features,
        "Transaction features",
    )

    validate_txid(
        network_features,
        "Network features",
    )

    transaction_features["txid"] = (
        transaction_features["txid"]
        .astype("string")
        .str.strip()
    )

    network_features["txid"] = (
        network_features["txid"]
        .astype("string")
        .str.strip()
    )

    # ---------------------------------------------------------------
    # Check for duplicate feature rows per txid
    # ---------------------------------------------------------------

    if transaction_features["txid"].duplicated().any():
        raise ValueError(
            "Transaction feature dataset contains duplicate txids."
        )

    if network_features["txid"].duplicated().any():
        raise ValueError(
            "Network feature dataset contains duplicate txids."
        )

    # ---------------------------------------------------------------
    # Combine using txid
    # ---------------------------------------------------------------

    combined = transaction_features.merge(
        network_features,
        on="txid",
        how="inner",
        suffixes=(
            "_transaction",
            "_network",
        ),
    )

    # ---------------------------------------------------------------
    # Verify matching coverage
    # ---------------------------------------------------------------

    transaction_txids = set(
        transaction_features["txid"]
    )

    network_txids = set(
        network_features["txid"]
    )

    common_txids = (
        transaction_txids
        & network_txids
    )

    transaction_only = (
        transaction_txids
        - network_txids
    )

    network_only = (
        network_txids
        - transaction_txids
    )

    # ---------------------------------------------------------------
    # Remove identifier from numeric ML matrix later, but keep txid
    # in this dataset for investigation traceability.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Create cross-domain interaction features
    # ---------------------------------------------------------------

    combined["fee_to_network_activity_ratio"] = (
        combined["fee"]
        / combined["network_events_per_transaction"]
        .replace(0, pd.NA)
    )

    combined["transaction_amount_per_network_event"] = (
        combined["input_amount"]
        / combined["network_events_per_transaction"]
        .replace(0, pd.NA)
    )

    combined["network_diversity_score"] = (
        combined["unique_destinations_per_source"]
        + combined["unique_destination_ports_per_source"]
        + combined["unique_countries_per_source"]
        + combined["unique_asns_per_source"]
    )

    combined["network_frequency_score"] = (
        combined["source_ip_frequency"]
        + combined["destination_ip_frequency"]
        + combined["src_dst_pair_frequency"]
    )

    # ---------------------------------------------------------------
    # Clean infinite values
    # ---------------------------------------------------------------

    combined = combined.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # ---------------------------------------------------------------
    # Sort and reset index
    # ---------------------------------------------------------------

    combined = combined.sort_values(
        by="txid"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined.to_csv(
            output_path,
            index=False,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("=" * 60)
    print("COMBINED FEATURE ENGINEERING")
    print("=" * 60)

    print(
        f"Transaction feature rows : "
        f"{len(transaction_features)}"
    )

    print(
        f"Network feature rows     : "
        f"{len(network_features)}"
    )

    print(
        f"Common TXIDs             : "
        f"{len(common_txids)}"
    )

    print(
        f"Transaction-only TXIDs   : "
        f"{len(transaction_only)}"
    )

    print(
        f"Network-only TXIDs       : "
        f"{len(network_only)}"
    )

    print(
        f"Combined rows            : "
        f"{len(combined)}"
    )

    print(
        f"Combined columns         : "
        f"{len(combined.columns)}"
    )

    print("\nValidation:")
    print("✓ Transaction features loaded")
    print("✓ Network features loaded")
    print("✓ TXIDs validated")
    print("✓ Duplicate TXIDs checked")
    print("✓ Feature sets combined using txid")
    print("✓ Cross-domain features generated")
    print("✓ Infinite values handled")

    return combined


def main():
    """Run combined feature generation."""

    output_path = Path(OUTPUT_FILE)

    combined = combine_features(
        transaction_path=TRANSACTION_FEATURES_FILE,
        network_path=NETWORK_FEATURES_FILE,
        output_path=output_path,
    )

    print("\nCombined feature dataset:")
    print(
        f"Rows    : {len(combined)}"
    )

    print(
        f"Columns : {len(combined.columns)}"
    )

    print("\nFirst 5 rows:")
    print(combined.head())

    print(
        f"\nSaved to: {output_path}"
    )

    print(
        "\nCombined feature engineering completed successfully."
    )


if __name__ == "__main__":
    main()
