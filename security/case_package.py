"""
Offline investigation case-package generation for Garud-Netra.

Creates a case package containing:

    CASE-XXXX/
    ├── report.json
    ├── evidence.json
    ├── transaction_graph.json
    └── signature.bin

The evidence.json file contains SHA-256 hashes of the supporting
report and transaction-graph artifacts.

The evidence document is then signed with ML-DSA-65 through the
PQC module.

Verification:
1. Verify the PQC signature over evidence.json.
2. Recalculate report.json hash.
3. Recalculate transaction_graph.json hash.
4. Compare both hashes with the signed evidence manifest.

This means modification of any signed case artifact is detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from security.pqc import (
    sign_case_evidence,
    verify_case_evidence,
)


# -------------------------------------------------------------------
# Input files
# -------------------------------------------------------------------

RISK_FILE = Path(
    "data/processed/risk_scores.csv"
)

EXPLANATION_FILE = Path(
    "data/processed/investigation_explanations.csv"
)

GRAPH_NODES_FILE = Path(
    "data/processed/graph_nodes.csv"
)

GRAPH_EDGES_FILE = Path(
    "data/processed/graph_edges.csv"
)


CASES_DIR = Path(
    "data/cases"
)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Required file is empty: {path}"
        )

    return df


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    """Write deterministic UTF-8 JSON."""

    path.write_text(
        json.dumps(
            data,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def find_case_record(
    txid: str,
    risk_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series | None]:
    """Find the requested investigation record and explanation."""

    txid = str(txid).strip()

    risk_matches = risk_df[
        risk_df["txid"].astype(str).str.strip()
        == txid
    ]

    if risk_matches.empty:
        raise ValueError(
            f"TXID not found in risk results: {txid}"
        )

    risk_record = risk_matches.iloc[0]

    explanation_matches = explanation_df[
        explanation_df["txid"].astype(str).str.strip()
        == txid
    ]

    explanation_record = (
        explanation_matches.iloc[0]
        if not explanation_matches.empty
        else None
    )

    return (
        risk_record,
        explanation_record,
    )


# -------------------------------------------------------------------
# Graph extraction
# -------------------------------------------------------------------

def build_transaction_graph_artifact(
    txid: str,
    graph_nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
) -> dict:
    """
    Extract the graph information associated with one transaction.
    """

    txid = str(txid).strip()

    transaction_node_id = (
        f"tx:{txid}"
    )

    # ---------------------------------------------------------------
    # Transaction node
    # ---------------------------------------------------------------

    transaction_nodes = graph_nodes[
        graph_nodes["node_id"].astype(str)
        == transaction_node_id
    ]

    nodes = []

    if not transaction_nodes.empty:

        for _, row in transaction_nodes.iterrows():

            nodes.append(
                {
                    "node_id": row["node_id"],
                    "node_type": row["node_type"],
                    "txid": row.get("txid"),
                    "degree": (
                        int(row["degree"])
                        if pd.notna(row.get("degree"))
                        else 0
                    ),
                    "in_degree": (
                        int(row["in_degree"])
                        if pd.notna(row.get("in_degree"))
                        else 0
                    ),
                    "out_degree": (
                        int(row["out_degree"])
                        if pd.notna(row.get("out_degree"))
                        else 0
                    ),
                    "degree_centrality": (
                        float(row["degree_centrality"])
                        if pd.notna(
                            row.get(
                                "degree_centrality"
                            )
                        )
                        else 0.0
                    ),
                    "connected_component": (
                        int(row["connected_component"])
                        if pd.notna(
                            row.get(
                                "connected_component"
                            )
                        )
                        else None
                    ),
                }
            )

    # ---------------------------------------------------------------
    # Edges connected to this transaction
    # ---------------------------------------------------------------

    related_edges = graph_edges[
        (
            graph_edges["source"].astype(str)
            == transaction_node_id
        )
        |
        (
            graph_edges["target"].astype(str)
            == transaction_node_id
        )
    ]

    edges = []

    related_node_ids = {
        transaction_node_id
    }

    for _, row in related_edges.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        related_node_ids.add(source)
        related_node_ids.add(target)

        edges.append(
            {
                "source": source,
                "target": target,
                "relationship": row.get(
                    "relationship"
                ),
            }
        )

    # ---------------------------------------------------------------
    # Include directly connected address nodes
    # ---------------------------------------------------------------

    related_nodes = graph_nodes[
        graph_nodes["node_id"].astype(str).isin(
            related_node_ids
        )
    ]

    for _, row in related_nodes.iterrows():

        node_id = str(row["node_id"])

        if node_id == transaction_node_id:
            continue

        nodes.append(
            {
                "node_id": node_id,
                "node_type": row.get(
                    "node_type"
                ),
                "address": row.get(
                    "address"
                ),
                "degree": (
                    int(row["degree"])
                    if pd.notna(row.get("degree"))
                    else 0
                ),
                "in_degree": (
                    int(row["in_degree"])
                    if pd.notna(row.get("in_degree"))
                    else 0
                ),
                "out_degree": (
                    int(row["out_degree"])
                    if pd.notna(row.get("out_degree"))
                    else 0
                ),
                "degree_centrality": (
                    float(
                        row["degree_centrality"]
                    )
                    if pd.notna(
                        row.get(
                            "degree_centrality"
                        )
                    )
                    else 0.0
                ),
                "connected_component": (
                    int(
                        row["connected_component"]
                    )
                    if pd.notna(
                        row.get(
                            "connected_component"
                        )
                    )
                    else None
                ),
            }
        )

    return {
        "transaction_node": transaction_node_id,
        "nodes": nodes,
        "edges": edges,
    }


# -------------------------------------------------------------------
# Case generation
# -------------------------------------------------------------------

def generate_case_package(
    case_id: str,
    txid: str,
    cases_root: Path = CASES_DIR,
) -> Path:
    """
    Generate a complete offline investigation case package.

    Returns
    -------
    Path
        Generated case directory.
    """

    risk_df = load_csv(
        RISK_FILE
    )

    explanation_df = load_csv(
        EXPLANATION_FILE
    )

    graph_nodes = load_csv(
        GRAPH_NODES_FILE
    )

    graph_edges = load_csv(
        GRAPH_EDGES_FILE
    )

    risk_record, explanation_record = (
        find_case_record(
            txid,
            risk_df,
            explanation_df,
        )
    )

    case_id = str(
        case_id
    ).strip()

    if not case_id:
        raise ValueError(
            "case_id cannot be empty."
        )

    txid = str(
        txid
    ).strip()

    case_directory = (
        cases_root / case_id
    )

    case_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Graph artifact
    # ---------------------------------------------------------------

    graph_artifact = (
        build_transaction_graph_artifact(
            txid=txid,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
    )

    graph_artifact.update(
        {
            "case_id": case_id,
            "txid": txid,
        }
    )

    graph_file = (
        case_directory
        / "transaction_graph.json"
    )

    write_json(
        graph_file,
        graph_artifact,
    )

    # ---------------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------------

    report = {
        "case_id": case_id,
        "generated_by": "Garud-Netra",
        "case_type": (
            "AI-assisted investigative lead"
        ),
        "txid": txid,
        "risk_score": float(
            risk_record["risk_score"]
        ),
        "priority": str(
            risk_record["priority"]
        ),
        "anomaly_score": float(
            risk_record["anomaly_score"]
        ),
        "anomaly_label": int(
            risk_record["anomaly_label"]
        ),
        "cluster_id": int(
            risk_record["cluster_id"]
        ),
        "explanation": (
            str(
                explanation_record[
                    "investigator_explanation"
                ]
            )
            if explanation_record is not None
            else str(
                risk_record["risk_reasons"]
            )
        ),
        "risk_reasons": str(
            risk_record["risk_reasons"]
        ),
    }

    report_file = (
        case_directory
        / "report.json"
    )

    write_json(
        report_file,
        report,
    )

    # ---------------------------------------------------------------
    # Signed evidence manifest
    # ---------------------------------------------------------------

    evidence = {
        "case_id": case_id,
        "algorithm": "ML-DSA-65",
        "txid": txid,
        "risk_score": float(
            risk_record["risk_score"]
        ),
        "priority": str(
            risk_record["priority"]
        ),
        "anomaly_score": float(
            risk_record["anomaly_score"]
        ),
        "anomaly_label": int(
            risk_record["anomaly_label"]
        ),
        "cluster_id": int(
            risk_record["cluster_id"]
        ),
        "risk_reasons": str(
            risk_record["risk_reasons"]
        ),
        "investigator_explanation": (
            str(
                explanation_record[
                    "investigator_explanation"
                ]
            )
            if explanation_record is not None
            else None
        ),
        "artifacts": {
            "report.json": {
                "sha256": sha256_file(
                    report_file
                )
            },
            "transaction_graph.json": {
                "sha256": sha256_file(
                    graph_file
                )
            },
        },
    }

    # sign_case_evidence writes the canonical evidence.json.
    signing_result = sign_case_evidence(
        case_directory=case_directory,
        evidence=evidence,
    )

    print("=" * 60)
    print("OFFLINE INVESTIGATION CASE PACKAGE")
    print("=" * 60)

    print(
        f"Case ID          : {case_id}"
    )

    print(
        f"TXID             : {txid}"
    )

    print(
        f"Risk score       : {report['risk_score']:.2f}"
    )

    print(
        f"Priority         : {report['priority']}"
    )

    print(
        "✓ report.json generated"
    )

    print(
        "✓ transaction_graph.json generated"
    )

    print(
        "✓ evidence.json generated"
    )

    print(
        "✓ ML-DSA-65 signature generated"
    )

    print(
        f"\nCase package: {case_directory}"
    )

    return case_directory


# -------------------------------------------------------------------
# Case verification
# -------------------------------------------------------------------

def verify_case_package(
    case_directory: str | Path,
) -> dict:
    """
    Verify both the PQC signature and supporting artifact hashes.
    """

    case_directory = Path(
        case_directory
    )

    evidence_file = (
        case_directory
        / "evidence.json"
    )

    report_file = (
        case_directory
        / "report.json"
    )

    graph_file = (
        case_directory
        / "transaction_graph.json"
    )

    # ---------------------------------------------------------------
    # Verify the PQC signature
    # ---------------------------------------------------------------

    pqc_result = verify_case_evidence(
        case_directory
    )

    # ---------------------------------------------------------------
    # Load signed evidence manifest
    # ---------------------------------------------------------------

    evidence = json.loads(
        evidence_file.read_text(
            encoding="utf-8"
        )
    )

    expected_report_hash = (
        evidence["artifacts"]["report.json"]["sha256"]
    )

    expected_graph_hash = (
        evidence["artifacts"][
            "transaction_graph.json"
        ]["sha256"]
    )

    actual_report_hash = (
        sha256_file(report_file)
    )

    actual_graph_hash = (
        sha256_file(graph_file)
    )

    report_valid = (
        actual_report_hash
        == expected_report_hash
    )

    graph_valid = (
        actual_graph_hash
        == expected_graph_hash
    )

    package_valid = (
        pqc_result["valid"]
        and report_valid
        and graph_valid
    )

    return {
        "valid": package_valid,
        "status": (
            "VALID"
            if package_valid
            else "INVALID"
        ),
        "pqc_signature_valid": (
            pqc_result["valid"]
        ),
        "report_integrity_valid": (
            report_valid
        ),
        "transaction_graph_integrity_valid": (
            graph_valid
        ),
        "case_id": evidence.get(
            "case_id"
        ),
        "txid": evidence.get(
            "txid"
        ),
    }


# -------------------------------------------------------------------
# Demonstration
# -------------------------------------------------------------------

def main():
    """
    Generate a case from the highest-risk investigation record
    and immediately verify it.
    """

    risk_df = load_csv(
        RISK_FILE
    )

    # Highest-risk record is the natural demonstration case.
    top_record = risk_df.sort_values(
        by="risk_score",
        ascending=False,
    ).iloc[0]

    txid = str(
        top_record["txid"]
    )

    case_directory = (
        generate_case_package(
            case_id="CASE-0001",
            txid=txid,
        )
    )

    verification = verify_case_package(
        case_directory
    )

    print("\nVerification:")
    print(
        f"Overall status              : "
        f"{verification['status']}"
    )

    print(
        f"PQC signature               : "
        f"{'VALID' if verification['pqc_signature_valid'] else 'INVALID'}"
    )

    print(
        f"Report integrity            : "
        f"{'VALID' if verification['report_integrity_valid'] else 'INVALID'}"
    )

    print(
        f"Transaction graph integrity : "
        f"{'VALID' if verification['transaction_graph_integrity_valid'] else 'INVALID'}"
    )


if __name__ == "__main__":
    main()
