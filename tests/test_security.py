"""
Automated security validation tests for Garud-Netra.
"""

from pathlib import Path

import pandas as pd
import pytest

from security.validation import (
    validate_csv_file,
    validate_offline_source,
)


def test_valid_csv_is_accepted(tmp_path):
    """A valid local CSV should pass validation."""

    root = tmp_path

    csv_file = (
        root / "valid.csv"
    )

    pd.DataFrame(
        {
            "txid": ["TX-001"],
            "value": [10.5],
        }
    ).to_csv(
        csv_file,
        index=False,
    )

    df = validate_csv_file(
        csv_file,
        root,
        required_columns=[
            "txid",
            "value",
        ],
    )

    assert len(df) == 1
    assert list(df.columns) == [
        "txid",
        "value",
    ]


def test_network_url_is_rejected():
    """Offline mode must reject network URLs."""

    with pytest.raises(ValueError):
        validate_offline_source(
            "https://example.com/data.csv"
        )


def test_non_csv_file_is_rejected(tmp_path):
    """Only CSV files should be accepted."""

    file_path = (
        tmp_path / "data.txt"
    )

    file_path.write_text(
        "test",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_csv_file(
            file_path,
            tmp_path,
        )


def test_path_outside_allowed_root_is_rejected(tmp_path):
    """Files outside the allowed root must be rejected."""

    allowed_root = (
        tmp_path / "allowed"
    )

    allowed_root.mkdir()

    outside_file = (
        tmp_path / "outside.csv"
    )

    pd.DataFrame(
        {"value": [1]}
    ).to_csv(
        outside_file,
        index=False,
    )

    with pytest.raises(ValueError):
        validate_csv_file(
            outside_file,
            allowed_root,
        )


def test_formula_injection_is_rejected(tmp_path):
    """Spreadsheet formula content must be rejected."""

    csv_file = (
        tmp_path / "malicious.csv"
    )

    pd.DataFrame(
        {
            "txid": [
                "=HYPERLINK('http://example.com')"
            ]
        }
    ).to_csv(
        csv_file,
        index=False,
    )

    with pytest.raises(ValueError):
        validate_csv_file(
            csv_file,
            tmp_path,
            required_columns=["txid"],
        )


def test_missing_required_column_is_rejected(tmp_path):
    """Missing schema columns must be rejected."""

    csv_file = (
        tmp_path / "missing_column.csv"
    )

    pd.DataFrame(
        {
            "value": [10]
        }
    ).to_csv(
        csv_file,
        index=False,
    )

    with pytest.raises(ValueError):
        validate_csv_file(
            csv_file,
            tmp_path,
            required_columns=[
                "txid",
                "value",
            ],
        )
