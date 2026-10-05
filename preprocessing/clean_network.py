"""
Network-event preprocessing for the Bitcoin AI Monitoring System.

This module:
- Loads network events through the ingestion layer
- Normalizes timestamps
- Normalizes transaction IDs
- Validates source/destination IP addresses
- Validates network ports
- Normalizes country codes
- Validates ASN values
- Removes duplicate network observations
- Saves cleaned data to data/processed/

The cleaned network data will later be correlated with
Bitcoin transaction data using txid and timestamp information.
"""

from pathlib import Path
from ipaddress import ip_address

import pandas as pd

from ingestion.csv_parser import load_network_events


REQUIRED_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "txid",
    "geo_country",
    "asn",
]


def normalize_text(value):
    """Strip leading and trailing whitespace from a value."""

    if pd.isna(value):
        return value

    return str(value).strip()


def normalize_txid(value):
    """Normalize a transaction ID."""

    if pd.isna(value):
        return value

    return str(value).strip()


def normalize_ip(value):
    """
    Validate and normalize an IPv4/IPv6 address.

    Returns the normalized textual representation.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return str(ip_address(value))
    except ValueError:
        return None


def normalize_country(value):
    """
    Normalize a country code/value.

    Empty values are represented as None.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if not value:
        return None

    return value


def validate_ports(df):
    """Validate that network ports are within the valid TCP/UDP range."""

    for column in ["src_port", "dst_port"]:

        invalid = (
            df[column].notna()
            & (
                (df[column] < 0)
                | (df[column] > 65535)
            )
        )

        if invalid.any():
            count = int(invalid.sum())

            raise ValueError(
                f"Column '{column}' contains "
                f"{count} invalid port value(s)."
            )


def validate_asn(df):
    """Validate that ASN values are non-negative."""

    invalid = (
        df["asn"].notna()
        & (df["asn"] < 0)
    )

    if invalid.any():
        count = int(invalid.sum())

        raise ValueError(
            f"Column 'asn' contains {count} negative value(s)."
        )


def clean_network(
    input_path,
    output_path=None,
):
    """
    Clean and normalize network-event data.

    Parameters
    ----------
    input_path:
        Path to the raw network-event CSV.

    output_path:
        Optional path for saving cleaned network events.

    Returns
    -------
    pandas.DataFrame
        Cleaned network-event DataFrame.
    """

    input_path = Path(input_path)

    # ---------------------------------------------------------------
    # Load using the ingestion module
    # ---------------------------------------------------------------

    df = load_network_events(input_path)

    original_rows = len(df)

    # ---------------------------------------------------------------
    # Validate schema
    # ---------------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Network dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------------
    # Normalize timestamps
    # ---------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------------
    # Normalize TXIDs
    # ---------------------------------------------------------------

    df["txid"] = df["txid"].apply(
        normalize_txid
    )

    # ---------------------------------------------------------------
    # Normalize IP addresses
    # ---------------------------------------------------------------

    df["src_ip"] = df["src_ip"].apply(
        normalize_ip
    )

    df["dst_ip"] = df["dst_ip"].apply(
        normalize_ip
    )

    # ---------------------------------------------------------------
    # Normalize country
    # ---------------------------------------------------------------

    df["geo_country"] = df["geo_country"].apply(
        normalize_country
    )

    # ---------------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------------

    for column in [
        "src_port",
        "dst_port",
        "asn",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # Remove rows missing critical network information
    # ---------------------------------------------------------------

    before_required_cleanup = len(df)

    df = df.dropna(
        subset=[
            "timestamp",
            "txid",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
        ]
    ).copy()

    removed_required = (
        before_required_cleanup - len(df)
    )

    # ---------------------------------------------------------------
    # Validate ports and ASN
    # ---------------------------------------------------------------

    validate_ports(df)

    validate_asn(df)

    # ---------------------------------------------------------------
    # Convert validated integer columns
    # ---------------------------------------------------------------

    df["src_port"] = df["src_port"].astype(int)
    df["dst_port"] = df["dst_port"].astype(int)

    if df["asn"].notna().all():
        df["asn"] = df["asn"].astype(int)

    # ---------------------------------------------------------------
    # Remove exact duplicate network observations
    #
    # We deliberately do NOT remove duplicate txids alone.
    # A real Bitcoin transaction can have multiple network
    # observations associated with it.
    # ---------------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "txid",
        ],
        keep="first",
    ).copy()

    duplicate_rows_removed = (
        before_duplicates - len(df)
    )

    # ---------------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------------

    df = df.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save processed data
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
    # Summary
    # ---------------------------------------------------------------

    cleaned_rows = len(df)

    print("=" * 60)
    print("NETWORK PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Input rows             : {original_rows}")
    print(f"Output rows            : {cleaned_rows}")
    print(f"Rows removed           : {original_rows - cleaned_rows}")
    print(f"Missing/invalid rows   : {removed_required}")
    print(
        f"Duplicate events removed: "
        f"{duplicate_rows_removed}"
    )

    print("\nValidation:")
    print("✓ Required columns validated")
    print("✓ Timestamps normalized")
    print("✓ TXIDs normalized")
    print("✓ Source IPs validated")
    print("✓ Destination IPs validated")
    print("✓ Ports validated")
    print("✓ Country values normalized")
    print("✓ ASN values validated")
    print("✓ Duplicate events removed")
    print("✓ Events sorted by timestamp")

    return df


if __name__ == "__main__":

    input_file = (
        "data/raw/synthetic_network_events.csv"
    )

    output_file = (
        "data/processed/clean_network_events.csv"
    )

    cleaned = clean_network(
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
        "\nNetwork preprocessing completed successfully."
    )
