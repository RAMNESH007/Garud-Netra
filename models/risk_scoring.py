"""
Risk scoring for the Bitcoin AI Monitoring System.

This module combines explainable evidence signals into a
0-100 investigative prioritization score.

The score is a prioritization aid. It does not establish
criminality or wrongdoing.
"""

from pathlib import Path

import pandas as pd


ANOMALY_FILE = "data/processed/anomaly_results.csv"
CLUSTER_FILE = "data/processed/cluster_results.csv"
GRAPH_FILE = "data/processed/graph_nodes.csv"

OUTPUT_FILE = "data/processed/risk_scores.csv"


def load_file(path):
    """Load a CSV file and verify that it exists and is non-empty."""

    path = Path(path)

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


def min_max_scale(series):
    """Scale a numeric series to the 0-100 range."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            0.0,
            index=series.index,
        )

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (values - minimum)
        / (maximum - minimum)
        * 100
    )


def build_risk_reasons(row):
    """Generate human-readable reasons from evidence signals."""

    reasons = []

    if row["anomaly_score"] >= 80:
        reasons.append(
            "high anomaly score"
        )
    elif row["anomaly_score"] >= 60:
        reasons.append(
            "elevated anomaly score"
        )

    if row["cluster_id"] == -1:
        reasons.append(
            "classified as DBSCAN noise"
        )

    if row["network_frequency_score"] >= row[
        "network_frequency_threshold"
    ]:
        reasons.append(
            "high network activity frequency"
        )

    if row["network_diversity_score"] >= row[
        "network_diversity_threshold"
    ]:
        reasons.append(
            "high network diversity"
        )

    if row["unique_destinations_per_source"] >= 3:
        reasons.append(
            "multiple destination IPs associated with source"
        )

    if row["unique_destination_ports_per_source"] >= 2:
        reasons.append(
            "multiple destination ports associated with source"
        )

    if row["fee_ratio"] >= row["fee_ratio_threshold"]:
        reasons.append(
            "elevated transaction fee ratio"
        )

    if row["input_output_ratio"] >= 1.05:
        reasons.append(
            "notable input/output amount difference"
        )

    if not reasons:
        reasons.append(
            "no high-priority evidence threshold triggered"
        )

    return "; ".join(reasons)


def calculate_risk_scores(
    anomaly_path,
    cluster_path,
    graph_path,
    output_path=None,
):
    """
    Combine anomaly, clustering, network and graph evidence
    into an explainable risk-prioritization result.
    """

    anomaly = load_file(anomaly_path)
    clusters = load_file(cluster_path)
    graph = load_file(graph_path)

    # ---------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------

    anomaly_required = [
        "txid",
        "anomaly_score",
        "anomaly_label",
    ]

    cluster_required = [
        "txid",
        "cluster_id",
    ]

    network_required = [
        "source_ip_frequency",
        "network_frequency_score",
        "network_diversity_score",
        "unique_destinations_per_source",
        "unique_destination_ports_per_source",
    ]

    transaction_required = [
        "fee_ratio",
        "input_output_ratio",
    ]

    for column in anomaly_required:
        if column not in anomaly.columns:
            raise ValueError(
                f"Anomaly results missing column: {column}"
            )

    for column in cluster_required:
        if column not in clusters.columns:
            raise ValueError(
                f"Cluster results missing column: {column}"
            )

    # ---------------------------------------------------------------
    # Merge anomaly + clustering information
    # ---------------------------------------------------------------

    results = anomaly.merge(
        clusters[
            [
                "txid",
                "cluster_id",
            ]
        ],
        on="txid",
        how="left",
    )

    # ---------------------------------------------------------------
    # Select compatible network/transaction evidence
    #
    # These columns come from the combined feature dataset already
    # present in anomaly_results.csv.
    # ---------------------------------------------------------------

    missing_evidence = [
        column
        for column in (
            network_required
            + transaction_required
        )
        if column not in results.columns
    ]

    if missing_evidence:
        raise ValueError(
            "Anomaly results missing evidence columns: "
            + ", ".join(missing_evidence)
        )

    # ---------------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------------

    evidence_columns = (
        network_required
        + transaction_required
    )

    for column in evidence_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    results["anomaly_score"] = pd.to_numeric(
        results["anomaly_score"],
        errors="coerce",
    )

    results["cluster_id"] = pd.to_numeric(
        results["cluster_id"],
        errors="coerce",
    ).fillna(-1).astype(int)

    # ---------------------------------------------------------------
    # Build normalized evidence signals
    # ---------------------------------------------------------------

    results["network_frequency_score_normalized"] = (
        min_max_scale(
            results["network_frequency_score"]
        )
    )

    results["network_diversity_score_normalized"] = (
        min_max_scale(
            results["network_diversity_score"]
        )
    )

    results["fee_ratio_normalized"] = (
        min_max_scale(
            results["fee_ratio"]
        )
    )

    results["input_output_difference_normalized"] = (
        min_max_scale(
            (
                results["input_output_ratio"]
                - 1
            ).abs()
        )
    )

    # ---------------------------------------------------------------
    # Calculate dataset-derived thresholds
    # ---------------------------------------------------------------

    network_frequency_threshold = (
        results["network_frequency_score"]
        .quantile(0.90)
    )

    network_diversity_threshold = (
        results["network_diversity_score"]
        .quantile(0.90)
    )

    fee_ratio_threshold = (
        results["fee_ratio"]
        .quantile(0.90)
    )

    results["network_frequency_threshold"] = (
        network_frequency_threshold
    )

    results["network_diversity_threshold"] = (
        network_diversity_threshold
    )

    results["fee_ratio_threshold"] = (
        fee_ratio_threshold
    )

    # ---------------------------------------------------------------
    # Weighted transparent score
    #
    # Weights sum to 100.
    # ---------------------------------------------------------------

    results["risk_score"] = (
        results["anomaly_score"] * 0.50
        + results["network_frequency_score_normalized"] * 0.15
        + results["network_diversity_score_normalized"] * 0.15
        + results["fee_ratio_normalized"] * 0.10
        + results["input_output_difference_normalized"] * 0.10
    )

    # ---------------------------------------------------------------
    # Cluster noise is recorded separately as evidence.
    #
    # We do not blindly add a large score simply because an entity
    # is DBSCAN noise.
    # ---------------------------------------------------------------

    results["cluster_noise_flag"] = (
        results["cluster_id"] == -1
    ).astype(int)

    # ---------------------------------------------------------------
    # Keep score within [0, 100]
    # ---------------------------------------------------------------

    results["risk_score"] = (
        results["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    # ---------------------------------------------------------------
    # Generate explanations
    # ---------------------------------------------------------------

    results["risk_reasons"] = (
        results.apply(
            build_risk_reasons,
            axis=1,
        )
    )

    # ---------------------------------------------------------------
    # Priority categories
    # ---------------------------------------------------------------

    results["priority"] = pd.cut(
        results["risk_score"],
        bins=[
            -0.01,
            40,
            70,
            100.01,
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
        ],
    )

    # ---------------------------------------------------------------
    # Graph information
    #
    # Aggregate graph metrics by txid for transaction nodes.
    # ---------------------------------------------------------------

    if "txid" in graph.columns:

        transaction_graph = graph[
            graph["node_type"] == "transaction"
        ].copy()

        graph_columns = [
            "txid",
            "degree",
            "degree_centrality",
            "connected_component",
        ]

        available_graph_columns = [
            column
            for column in graph_columns
            if column in transaction_graph.columns
        ]

        if len(available_graph_columns) > 1:

            results = results.merge(
                transaction_graph[
                    available_graph_columns
                ],
                on="txid",
                how="left",
            )

    # ---------------------------------------------------------------
    # Sort by priority score
    # ---------------------------------------------------------------

    results = results.sort_values(
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

        results.to_csv(
            output_path,
            index=False,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("=" * 60)
    print("INVESTIGATIVE RISK SCORING")
    print("=" * 60)

    print(
        f"Input records       : {len(results)}"
    )

    print(
        f"Minimum risk score  : "
        f"{results['risk_score'].min():.2f}"
    )

    print(
        f"Maximum risk score  : "
        f"{results['risk_score'].max():.2f}"
    )

    print(
        f"High priority       : "
        f"{(results['priority'] == 'HIGH').sum()}"
    )

    print(
        f"Medium priority     : "
        f"{(results['priority'] == 'MEDIUM').sum()}"
    )

    print(
        f"Low priority        : "
        f"{(results['priority'] == 'LOW').sum()}"
    )

    print("\nValidation:")
    print("✓ Anomaly evidence loaded")
    print("✓ Cluster evidence loaded")
    print("✓ Network evidence validated")
    print("✓ Transaction evidence validated")
    print("✓ Evidence normalized")
    print("✓ Transparent weighted score calculated")
    print("✓ Risk reasons generated")
    print("✓ Priority categories generated")
    print("✓ Graph metrics attached where available")

    if output_path is not None:
        print(
            f"\nSaved results to: {output_path}"
        )

    return results


def main():
    """Run investigative risk scoring."""

    results = calculate_risk_scores(
        anomaly_path=ANOMALY_FILE,
        cluster_path=CLUSTER_FILE,
        graph_path=GRAPH_FILE,
        output_path=OUTPUT_FILE,
    )

    print("\nTop 10 prioritized records:")

    print(
        results[
            [
                "txid",
                "risk_score",
                "priority",
                "risk_reasons",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print(
        "\nRisk scoring completed successfully."
    )


if __name__ == "__main__":
    main()
