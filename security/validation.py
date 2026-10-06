"""
Security validation utilities for the Bitcoin AI Monitoring System.

This module provides defensive validation for local/offline data
processing.

Security checks include:
- safe local path validation
- allowed file extensions
- file size limits
- CSV structure limits
- required-column validation
- dangerous spreadsheet-formula detection
- malformed CSV handling
- basic offline-input validation

The module does not make network calls.
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


# -------------------------------------------------------------------
# Security limits
# -------------------------------------------------------------------

MAX_FILE_SIZE_MB = 50
MAX_CSV_ROWS = 1_000_000
MAX_CSV_COLUMNS = 100


# -------------------------------------------------------------------
# Path validation
# -------------------------------------------------------------------

def validate_local_path(
    file_path,
    allowed_root,
):
    """
    Validate that a file exists and remains inside an allowed
    local directory.

    This protects against path traversal and unintended access
    outside the application data directory.
    """

    file_path = Path(file_path).expanduser()
    allowed_root = Path(allowed_root).expanduser()

    if not allowed_root.exists():
        raise FileNotFoundError(
            f"Allowed root does not exist: {allowed_root}"
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file: {file_path}"
        )

    resolved_file = file_path.resolve()
    resolved_root = allowed_root.resolve()

    try:
        resolved_file.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ValueError(
            "Path is outside the allowed application directory: "
            f"{file_path}"
        ) from exc

    return resolved_file


# -------------------------------------------------------------------
# File extension validation
# -------------------------------------------------------------------

def validate_csv_extension(file_path):
    """
    Ensure the supplied file uses a CSV extension.
    """

    suffix = Path(file_path).suffix.lower()

    if suffix != ".csv":
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Only .csv files are allowed."
        )


# -------------------------------------------------------------------
# File size validation
# -------------------------------------------------------------------

def validate_file_size(
    file_path,
    max_size_mb=MAX_FILE_SIZE_MB,
):
    """
    Reject files larger than the configured size limit.
    """

    file_path = Path(file_path)

    size_bytes = file_path.stat().st_size
    max_size_bytes = (
        max_size_mb * 1024 * 1024
    )

    if size_bytes > max_size_bytes:
        raise ValueError(
            f"File is too large: "
            f"{size_bytes / (1024 * 1024):.2f} MB. "
            f"Maximum allowed is {max_size_mb} MB."
        )

    return size_bytes


# -------------------------------------------------------------------
# CSV formula-injection detection
# -------------------------------------------------------------------

FORMULA_PREFIXES = (
    "=",
    "+",
    "-",
    "@",
)


def contains_formula_injection(value):
    """
    Detect values that begin with common spreadsheet formula
    prefixes.

    This is a detection function only. It does not modify data.
    """

    if pd.isna(value):
        return False

    value = str(value).lstrip()

    return value.startswith(
        FORMULA_PREFIXES
    )


def find_formula_injection_cells(df):
    """
    Find cells that could become spreadsheet formulas when the
    CSV is opened by a spreadsheet application.

    Returns
    -------
    list[tuple]
        Tuples containing (row_index, column_name).
    """

    dangerous_cells = []

    for column in df.columns:

        mask = df[column].apply(
            contains_formula_injection
        )

        for row_index in df.index[mask]:
            dangerous_cells.append(
                (
                    row_index,
                    column,
                )
            )

    return dangerous_cells


# -------------------------------------------------------------------
# CSV structure validation
# -------------------------------------------------------------------

def validate_csv_structure(
    file_path,
    required_columns=None,
    max_rows=MAX_CSV_ROWS,
    max_columns=MAX_CSV_COLUMNS,
):
    """
    Safely inspect a CSV and validate basic structural limits.

    Returns
    -------
    pandas.DataFrame
        Loaded CSV data.
    """

    required_columns = (
        required_columns or []
    )

    validate_csv_extension(
        file_path
    )

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            f"Malformed or unreadable CSV: {file_path}"
        ) from exc

    # ---------------------------------------------------------------
    # Resource limits
    # ---------------------------------------------------------------

    if len(df) > max_rows:
        raise ValueError(
            f"CSV contains {len(df)} rows. "
            f"Maximum allowed is {max_rows}."
        )

    if len(df.columns) > max_columns:
        raise ValueError(
            f"CSV contains {len(df.columns)} columns. "
            f"Maximum allowed is {max_columns}."
        )

    # ---------------------------------------------------------------
    # Required columns
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

    return df


# -------------------------------------------------------------------
# Offline source validation
# -------------------------------------------------------------------

def validate_offline_source(source):
    """
    Validate that a data source refers to a local file rather than
    an HTTP/HTTPS/network resource.

    The final application is intended to operate offline.
    """

    source = str(source).strip()

    parsed = urlparse(source)

    if parsed.scheme in {
        "http",
        "https",
        "ftp",
        "ftps",
    }:
        raise ValueError(
            "Network URLs are not allowed in offline mode: "
            f"{source}"
        )

    return True


# -------------------------------------------------------------------
# Complete file validation
# -------------------------------------------------------------------

def validate_csv_file(
    file_path,
    allowed_root,
    required_columns=None,
):
    """
    Run the complete local/offline CSV validation process.

    Returns
    -------
    pandas.DataFrame
        Validated CSV data.
    """

    validate_offline_source(
        file_path
    )

    safe_path = validate_local_path(
        file_path,
        allowed_root,
    )

    validate_csv_extension(
        safe_path
    )

    size_bytes = validate_file_size(
        safe_path
    )

    df = validate_csv_structure(
        safe_path,
        required_columns=required_columns,
    )

    formula_cells = find_formula_injection_cells(
        df
    )

    if formula_cells:
        raise ValueError(
            "Potential CSV/spreadsheet formula injection detected "
            f"in {len(formula_cells)} cell(s)."
        )

    print("=" * 60)
    print("SECURITY VALIDATION")
    print("=" * 60)

    print(
        f"Validated file      : {safe_path}"
    )

    print(
        f"File size           : "
        f"{size_bytes / 1024:.2f} KB"
    )

    print(
        f"Rows                : {len(df)}"
    )

    print(
        f"Columns             : {len(df.columns)}"
    )

    print("\nValidation:")
    print("✓ Local path validated")
    print("✓ File extension validated")
    print("✓ File size validated")
    print("✓ CSV parsed safely")
    print("✓ Row/column limits validated")

    if required_columns:
        print("✓ Required columns validated")

    print("✓ No formula-injection cells detected")
    print("✓ Offline source validated")

    return df


# -------------------------------------------------------------------
# Test / demonstration
# -------------------------------------------------------------------

if __name__ == "__main__":

    project_root = Path(
        "."
    ).resolve()

    transaction_file = Path(
        "data/raw/synthetic_transactions.csv"
    )

    required_columns = [
        "timestamp",
        "txid",
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts",
        "fee",
        "script_type",
    ]

    validate_csv_file(
        file_path=transaction_file,
        allowed_root=project_root,
        required_columns=required_columns,
    )

    print(
        "\nSecurity validation completed successfully."
    )
