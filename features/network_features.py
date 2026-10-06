"""
Network feature engineering for the Bitcoin AI Monitoring System.

This module creates explainable network-level features from the
correlated transaction + network dataset.

The generated features are intended for:
- anomaly detection
- risk scoring
- investigation explanations
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = "data/processed/correlated_data.csv"
OUTPUT_FILE = "data/processed/network_features.csv"


def create_network_features(df):
    """
    Create explainable network-level features.

    Parameters
    ----------
    df:
        Correlated transaction/network DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing network features.
    """

    required_columns = [
        "txid",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "geo_country",
        "asn",
        "network_timestamp",
        "absolute_time_difference_seconds",
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

    features["txid"] = (
        df["txid"]
        .astype("string")
        .str.strip()
    )

    # ---------------------------------------------------------------
    # Basic network attributes
    # ---------------------------------------------------------------

    features["src_port"] = pd.to_numeric(
        df["src_port"],
        errors="coerce",
    )

    features["dst_port"] = pd.to_numeric(
        df["dst_port"],
        errors="coerce",
    )

    features["asn"] = pd.to_numeric(
        df["asn"],
        errors="coerce",
    )

    # ---------------------------------------------------------------
    # Source IP frequency
    # ---------------------------------------------------------------

    source_frequency = (
        df["src_ip"]
        .value_counts(dropna=False)
    )

    features["source_ip_frequency"] = (
        df["src_ip"]
        .map(source_frequency)
    )

    # ---------------------------------------------------------------
    # Destination IP frequency
    # ---------------------------------------------------------------

    destination_frequency = (
        df["dst_ip"]
        .value_counts(dropna=False)
    )

    features["destination_ip_frequency"] = (
        df["dst_ip"]
        .map(destination_frequency)
    )

    # ---------------------------------------------------------------
    # Unique destinations contacted by each source
    # ---------------------------------------------------------------

    unique_destinations = (
        df.groupby("src_ip")["dst_ip"]
        .nunique()
    )

    features["unique_destinations_per_source"] = (
        df["src_ip"].map(unique_destinations)
    )

    # ---------------------------------------------------------------
    # Unique sources communicating with each destination
    # ---------------------------------------------------------------

    unique_sources = (
        df.groupby("dst_ip")["src_ip"]
        .nunique()
    )

    features["unique_sources_per_destination"] = (
        df["dst_ip"].map(unique_sources)
    )

    # ---------------------------------------------------------------
    # Source/destination pair frequency
    # ---------------------------------------------------------------

    pair_frequency = (
        df.groupby(
            ["src_ip", "dst_ip"]
        )
        .size()
    )

    pair_keys = list(
        zip(
            df["src_ip"],
            df["dst_ip"],
        )
    )

    features["src_dst_pair_frequency"] = [
        pair_frequency.get(key, 0)
        for key in pair_keys
    ]

    # ---------------------------------------------------------------
    # Port frequency
    # ---------------------------------------------------------------

    source_port_frequency = (
        df["src_port"]
        .value_counts(dropna=False)
    )

    destination_port_frequency = (
        df["dst_port"]
        .value_counts(dropna=False)
    )

    features["source_port_frequency"] = (
        df["src_port"]
        .map(source_port_frequency)
    )

    features["destination_port_frequency"] = (
        df["dst_port"]
        .map(destination_port_frequency)
    )

    # ---------------------------------------------------------------
    # Rare port indicator
    #
    # A port is considered rare when it appears in <= 1% of
    # observations. This is data-driven rather than hard-coded.
    # ---------------------------------------------------------------

    total_rows = len(df)

    rare_port_threshold = max(
        1,
        int(total_rows * 0.01),
    )

    features["is_rare_destination_port"] = (
        features["destination_port_frequency"]
        <= rare_port_threshold
    ).astype(int)

    features["is_rare_source_port"] = (
        features["source_port_frequency"]
        <= rare_port_threshold
    ).astype(int)

    # ---------------------------------------------------------------
    # Country frequency
    # ---------------------------------------------------------------

    country_frequency = (
        df["geo_country"]
        .value_counts(dropna=False)
    )

    features["country_frequency"] = (
        df["geo_country"]
        .map(country_frequency)
    )

    # ---------------------------------------------------------------
    # ASN frequency
    # ---------------------------------------------------------------

    asn_frequency = (
        df["asn"]
        .value_counts(dropna=False)
    )

    features["asn_frequency"] = (
        df["asn"]
        .map(asn_frequency)
    )

    # ---------------------------------------------------------------
    # Transaction/network timing
    # ---------------------------------------------------------------

    features["network_transaction_time_gap_seconds"] = (
        pd.to_numeric(
            df["absolute_time_difference_seconds"],
            errors="coerce",
        )
    )

    # A simple flag for unusually large timing gaps.
    #
    # The threshold is derived from the dataset rather than
    # hard-coded.
    timing_gap = (
        features["network_transaction_time_gap_seconds"]
    )

    if timing_gap.notna().any():

        timing_threshold = timing_gap.quantile(
            0.95
        )

        features["large_time_gap_flag"] = (
            timing_gap > timing_threshold
        ).astype(int)

    else:

        features["large_time_gap_flag"] = 0

    # ---------------------------------------------------------------
    # Network activity per transaction
    # ---------------------------------------------------------------

    transaction_network_counts = (
        df.groupby("txid")
        .size()
    )

    features["network_events_per_transaction"] = (
        features["txid"]
        .map(transaction_network_counts)
    )

    # ---------------------------------------------------------------
    # Connection diversity
    # ---------------------------------------------------------------

    ports_per_source = (
        df.groupby("src_ip")["dst_port"]
        .nunique()
    )

    features["unique_destination_ports_per_source"] = (
        df["src_ip"].map(ports_per_source)
    )

    # ---------------------------------------------------------------
    # Country / ASN diversity per source
    # ---------------------------------------------------------------

    countries_per_source = (
        df.groupby("src_ip")["geo_country"]
        .nunique()
    )

    asns_per_source = (
        df.groupby("src_ip")["asn"]
        .nunique()
    )

    features["unique_countries_per_source"] = (
        df["src_ip"].map(countries_per_source)
    )

    features["unique_asns_per_source"] = (
        df["src_ip"].map(asns_per_source)
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
    """Run network feature engineering."""

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    feature_df = create_network_features(
        df
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 60)
    print("NETWORK FEATURE ENGINEERING")
    print("=" * 60)

    print(
        f"Input rows     : {len(df)}"
    )

    print(
        f"Output rows    : {len(feature_df)}"
    )

    print(
        f"Feature count  : {len(feature_df.columns)}"
    )

    print("\nGenerated features:")

    for column in feature_df.columns:
        print(f"✓ {column}")

    print(
        f"\nSaved to: {output_path}"
    )

    print(
        "\nNetwork feature engineering completed successfully."
    )


if __name__ == "__main__":
    main()
