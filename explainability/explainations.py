"""
Explainability module for the Bitcoin AI Monitoring System.

This module converts anomaly, clustering, graph and risk-scoring
signals into structured, human-readable investigation explanations.

The output is intended for investigators and dashboard display.
It describes evidence and model signals without asserting criminality.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = "data/processed/risk_scores.csv"
OUTPUT_FILE = "data/processed/investigation_explanations.csv"


def load_risk_results(input_path):
    """Load and validate risk-scoring results."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Risk results not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(
            "Risk results file is empty."
        )

    required_columns = [
        "txid",
        "risk_score",
        "priority",
        "risk_reasons",
        "anomaly_score",
        "anomaly_label",
        "cluster_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Risk results are missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def split_reasons(reason_text):
    """Convert the semicolon-separated reason field into a list."""

    if pd.isna(reason_text):
        return []

    reason_text = str(reason_text).strip()

    if not reason_text:
        return []

    return [
        reason.strip()
        for reason in reason_text.split(";")
        if reason.strip()
    ]


def build_evidence_summary(row):
    """
    Build a structured evidence summary from one investigation row.
    """

    evidence = []

    evidence.append(
        f"Anomaly score: {row['anomaly_score']:.2f}"
    )

    evidence.append(
        f"Risk score: {row['risk_score']:.2f}"
    )

    evidence.append(
        f"Priority: {row['priority']}"
    )

    evidence.append(
        f"Cluster ID: {int(row['cluster_id'])}"
    )

    if row["anomaly_label"] == 1:
        evidence.append(
            "The anomaly detector classified this record as anomalous."
        )
    else:
        evidence.append(
            "The anomaly detector classified this record as normal."
        )

    if int(row["cluster_id"]) == -1:
        evidence.append(
            "DBSCAN classified this record as noise rather than "
            "assigning it to a dense behavioral cluster."
        )
    else:
        evidence.append(
            f"The record belongs to behavioral cluster "
            f"{int(row['cluster_id'])}."
        )

    return " ".join(evidence)


def build_investigation_explanation(
    row,
    case_number,
):
    """Create a complete investigator-readable explanation."""

    reasons = split_reasons(
        row["risk_reasons"]
    )

    explanation_parts = []

    for index, reason in enumerate(
        reasons,
        start=1,
    ):
        explanation_parts.append(
            f"{index}. {reason}."
        )

    explanation_text = " ".join(
        explanation_parts
    )

    evidence_summary = build_evidence_summary(
        row
    )

    return {
        "investigation_id": (
            f"CASE-{case_number:04d}"
        ),
        "txid": row["txid"],
        "risk_score": round(
            float(row["risk_score"]),
            2,
        ),
        "priority": str(
            row["priority"]
        ),
        "anomaly_score": round(
            float(row["anomaly_score"]),
            2,
        ),
        "anomaly_label": int(
            row["anomaly_label"]
        ),
        "cluster_id": int(
            row["cluster_id"]
        ),
        "risk_reasons": str(
            row["risk_reasons"]
        ),
        "evidence_summary": evidence_summary,
        "investigator_explanation": (
            explanation_text
        ),
    }


def generate_explanations(
    input_path,
    output_path=None,
):
    """
    Generate explanations for all investigation records.
    """

    df = load_risk_results(
        input_path
    )

    # ---------------------------------------------------------------
    # Normalize numeric fields
    # ---------------------------------------------------------------

    for column in [
        "risk_score",
        "anomaly_score",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["anomaly_label"] = pd.to_numeric(
        df["anomaly_label"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["cluster_id"] = pd.to_numeric(
        df["cluster_id"],
        errors="coerce",
    ).fillna(-1).astype(int)

    # ---------------------------------------------------------------
    # Generate investigation records
    # ---------------------------------------------------------------

    records = []

    for case_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        records.append(
            build_investigation_explanation(
                row,
                case_number,
            )
        )

    explanations = pd.DataFrame(
        records
    )

    # ---------------------------------------------------------------
    # Sort by risk priority
    # ---------------------------------------------------------------

    explanations = explanations.sort_values(
        by="risk_score",
        ascending=False,
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        explanations.to_csv(
            output_path,
            index=False,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    high_priority = (
        explanations["priority"] == "HIGH"
    ).sum()

    medium_priority = (
        explanations["priority"] == "MEDIUM"
    ).sum()

    low_priority = (
        explanations["priority"] == "LOW"
    ).sum()

    print("=" * 60)
    print("INVESTIGATION EXPLAINABILITY")
    print("=" * 60)

    print(
        f"Investigation records : {len(explanations)}"
    )

    print(
        f"HIGH priority         : {high_priority}"
    )

    print(
        f"MEDIUM priority       : {medium_priority}"
    )

    print(
        f"LOW priority          : {low_priority}"
    )

    print("\nValidation:")
    print("✓ Risk results loaded")
    print("✓ Required evidence fields validated")
    print("✓ Numerical fields normalized")
    print("✓ Risk reasons parsed")
    print("✓ Investigator explanations generated")
    print("✓ Investigation IDs generated")
    print("✓ Results sorted by risk score")

    if output_path is not None:
        print(
            f"\nSaved explanations to: {output_path}"
        )

    return explanations


def main():
    """Run explainability generation."""

    explanations = generate_explanations(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
    )

    print("\nTop 5 investigation records:")

    print(
        explanations[
            [
                "investigation_id",
                "txid",
                "risk_score",
                "priority",
                "investigator_explanation",
            ]
        ].head(5).to_string(
            index=False
        )
    )

    print(
        "\nExplainability generation completed successfully."
    )


if __name__ == "__main__":
    main()
