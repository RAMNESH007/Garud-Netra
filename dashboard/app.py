"""
Garud-Netra Investigator Dashboard.

Offline Streamlit dashboard for exploring:
- investigation priorities
- anomaly scores
- risk scores
- clustering results
- graph statistics
- investigator explanations

No external APIs or network services are used.
"""

import json
import sys
from pathlib import Path

# -------------------------------------------------------------------
# Project root / local package path
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from security.case_package import verify_case_package


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RISK_FILE = (
    PROCESSED_DIR / "risk_scores.csv"
)

EXPLANATION_FILE = (
    PROCESSED_DIR / "investigation_explanations.csv"
)

CLUSTER_FILE = (
    PROCESSED_DIR / "cluster_results.csv"
)

GRAPH_NODES_FILE = (
    PROCESSED_DIR / "graph_nodes.csv"
)

GRAPH_EDGES_FILE = (
    PROCESSED_DIR / "graph_edges.csv"
)

CASES_DIR = (
    PROJECT_ROOT / "data" / "cases"
)
# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Garud-Netra",
    page_icon="🔎",
    layout="wide",
)


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

@st.cache_data
def load_csv(path):
    """Load a local CSV file."""

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_all_data():
    """Load all locally generated investigation artifacts."""

    risk = load_csv(RISK_FILE)
    explanations = load_csv(
        EXPLANATION_FILE
    )
    clusters = load_csv(
        CLUSTER_FILE
    )
    graph_nodes = load_csv(
        GRAPH_NODES_FILE
    )
    graph_edges = load_csv(
        GRAPH_EDGES_FILE
    )

    return (
        risk,
        explanations,
        clusters,
        graph_nodes,
        graph_edges,
    )


(
    risk_df,
    explanation_df,
    cluster_df,
    graph_nodes_df,
    graph_edges_df,
) = load_all_data()

def find_case_for_txid(txid):
    """
    Find the local offline case package associated with a TXID.

    Returns
    -------
    Path | None
        Matching case directory, or None when no case exists.
    """

    if not CASES_DIR.is_dir():
        return None

    txid = str(txid).strip()

    for case_directory in sorted(
        CASES_DIR.iterdir()
    ):

        if not case_directory.is_dir():
            continue

        evidence_file = (
            case_directory / "evidence.json"
        )

        if not evidence_file.is_file():
            continue

        try:
            evidence = json.loads(
                evidence_file.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            continue

        if str(
            evidence.get("txid", "")
        ).strip() == txid:

            return case_directory

    return None

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("🔎 Garud-Netra")
st.subheader(
    "AI-Powered Bitcoin Transaction Monitoring & Investigation"
)

st.caption(
    "Offline investigative dashboard • "
    "Model outputs are evidence signals, not determinations of wrongdoing."
)


# -------------------------------------------------------------------
# Data availability check
# -------------------------------------------------------------------

required_data = {
    "Risk Scores": risk_df,
    "Investigation Explanations": explanation_df,
    "Cluster Results": cluster_df,
    "Graph Nodes": graph_nodes_df,
    "Graph Edges": graph_edges_df,
}


missing_data = [
    name
    for name, data in required_data.items()
    if data.empty
]

if missing_data:

    st.error(
        "Required processed data is missing."
    )

    st.write(
        "Missing datasets:"
    )

    for item in missing_data:
        st.write(f"- {item}")

    st.info(
        "Run the pipeline stages that generate "
        "the required files before launching the dashboard."
    )

    st.stop()


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------

st.sidebar.header("Investigation Filters")

available_priorities = sorted(
    risk_df["priority"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_priorities = st.sidebar.multiselect(
    "Priority",
    options=available_priorities,
    default=available_priorities,
)

min_risk = float(
    risk_df["risk_score"].min()
)

max_risk = float(
    risk_df["risk_score"].max()
)

risk_range = st.sidebar.slider(
    "Risk score range",
    min_value=0.0,
    max_value=100.0,
    value=(
        round(min_risk, 2),
        round(max_risk, 2),
    ),
)

anomaly_only = st.sidebar.checkbox(
    "Show anomaly-labelled records only"
)

search_txid = st.sidebar.text_input(
    "Search TXID"
).strip()


# -------------------------------------------------------------------
# Apply filters
# -------------------------------------------------------------------

filtered_df = risk_df.copy()

filtered_df = filtered_df[
    filtered_df["priority"].isin(
        selected_priorities
    )
]

filtered_df = filtered_df[
    filtered_df["risk_score"].between(
        risk_range[0],
        risk_range[1],
    )
]

if anomaly_only:

    filtered_df = filtered_df[
        filtered_df["anomaly_label"] == 1
    ]

if search_txid:

    filtered_df = filtered_df[
        filtered_df["txid"]
        .astype(str)
        .str.contains(
            search_txid,
            case=False,
            na=False,
        )
    ]


# -------------------------------------------------------------------
# Overview metrics
# -------------------------------------------------------------------

st.header("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Investigations",
        len(filtered_df),
    )

with col2:

    high_count = int(
        (
            filtered_df["priority"]
            == "HIGH"
        ).sum()
    )

    st.metric(
        "High Priority",
        high_count,
    )

with col3:

    anomaly_count = int(
        filtered_df["anomaly_label"].sum()
    )

    st.metric(
        "Anomalies",
        anomaly_count,
    )

with col4:

    st.metric(
        "Average Risk",
        f"{filtered_df['risk_score'].mean():.2f}"
        if not filtered_df.empty
        else "0.00",
    )


if filtered_df.empty:

    st.warning(
        "No records match the current filters."
    )

    st.stop()


# -------------------------------------------------------------------
# Risk distribution
# -------------------------------------------------------------------

st.header("Risk Analysis")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:

    fig_risk = px.histogram(
        filtered_df,
        x="risk_score",
        nbins=20,
        title="Risk Score Distribution",
        labels={
            "risk_score": "Risk Score"
        },
    )

    fig_risk.update_layout(
        yaxis_title="Investigations"
    )

    st.plotly_chart(
        fig_risk,
        width="stretch",
    )


with risk_col2:

    priority_counts = (
        filtered_df["priority"]
        .value_counts()
        .reset_index()
    )

    priority_counts.columns = [
        "priority",
        "count",
    ]

    fig_priority = px.pie(
        priority_counts,
        names="priority",
        values="count",
        title="Priority Distribution",
    )

    st.plotly_chart(
        fig_priority,
        width="stretch",
    )


# -------------------------------------------------------------------
# Top investigations
# -------------------------------------------------------------------

st.header("Top Investigative Leads")

display_columns = [
    "txid",
    "risk_score",
    "priority",
    "anomaly_score",
    "anomaly_label",
]

available_display_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

top_records = (
    filtered_df[
        available_display_columns
    ]
    .sort_values(
        "risk_score",
        ascending=False,
    )
    .head(20)
)

st.dataframe(
    top_records,
    width="stretch",
    hide_index=True,
)


# -------------------------------------------------------------------
# Investigation detail
# -------------------------------------------------------------------

st.header("Investigation Detail")

available_txids = (
    filtered_df["txid"]
    .astype(str)
    .tolist()
)

selected_txid = st.selectbox(
    "Select a transaction",
    options=available_txids,
)


selected_record = filtered_df[
    filtered_df["txid"].astype(str)
    == selected_txid
].iloc[0]


detail_col1, detail_col2, detail_col3 = (
    st.columns(3)
)

with detail_col1:

    st.metric(
        "Risk Score",
        f"{selected_record['risk_score']:.2f}",
    )

with detail_col2:

    st.metric(
        "Anomaly Score",
        f"{selected_record['anomaly_score']:.2f}",
    )

with detail_col3:

    st.metric(
        "Priority",
        str(selected_record["priority"]),
    )


st.write(
    f"**TXID:** `{selected_txid}`"
)


# -------------------------------------------------------------------
# Explanation
# -------------------------------------------------------------------

matching_explanation = (
    explanation_df[
        explanation_df["txid"].astype(str)
        == selected_txid
    ]
)

if not matching_explanation.empty:

    explanation = (
        matching_explanation.iloc[0]
    )

    st.subheader(
        "Why was this record surfaced?"
    )

    st.info(
        explanation[
            "evidence_summary"
        ]
    )

    st.write(
        explanation[
            "investigator_explanation"
        ]
    )


# -------------------------------------------------------------------
# Evidence fields
# -------------------------------------------------------------------

st.subheader("Evidence Signals")

signal_columns = [
    "risk_score",
    "anomaly_score",
    "anomaly_label",
    "cluster_id",
    "network_frequency_score",
    "network_diversity_score",
    "source_ip_frequency",
    "destination_ip_frequency",
    "unique_destinations_per_source",
    "unique_destination_ports_per_source",
    "fee_ratio",
    "input_output_ratio",
    "network_events_per_transaction",
]

available_signal_columns = [
    column
    for column in signal_columns
    if column in selected_record.index
]

signal_data = pd.DataFrame(
    {
        "Signal": available_signal_columns,
        "Value": [
            selected_record[column]
            for column in available_signal_columns
        ],
    }
)

st.dataframe(
    signal_data,
    width="stretch",
    hide_index=True,
)


# -------------------------------------------------------------------
# Clustering
# -------------------------------------------------------------------

st.header("Entity Clustering")

cluster_distribution = (
    cluster_df["cluster_id"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_distribution.columns = [
    "cluster_id",
    "count",
]

fig_clusters = px.bar(
    cluster_distribution,
    x="cluster_id",
    y="count",
    title="Cluster Distribution",
)

st.plotly_chart(
    fig_clusters,
    width="stretch",
)

selected_cluster = selected_record.get(
    "cluster_id",
    None,
)

if selected_cluster is not None:

    st.write(
        f"Selected record cluster: "
        f"**{selected_cluster}**"
    )

    if int(selected_cluster) == -1:
        st.warning(
            "This record was classified as DBSCAN noise."
        )


# -------------------------------------------------------------------
# Graph statistics
# -------------------------------------------------------------------

st.header("Transaction Graph")

graph_col1, graph_col2, graph_col3 = (
    st.columns(3)
)

with graph_col1:
    st.metric(
        "Graph Nodes",
        len(graph_nodes_df),
    )

with graph_col2:
    st.metric(
        "Graph Edges",
        len(graph_edges_df),
    )

with graph_col3:

    if "connected_component" in graph_nodes_df.columns:

        component_count = (
            graph_nodes_df[
                "connected_component"
            ]
            .nunique()
        )

    else:

        component_count = 0

    st.metric(
        "Components",
        component_count,
    )


# -------------------------------------------------------------------
# Graph node information for selected transaction
# -------------------------------------------------------------------

if (
    "txid" in graph_nodes_df.columns
    and not graph_nodes_df.empty
):

    selected_graph_nodes = (
        graph_nodes_df[
            graph_nodes_df["txid"]
            .astype(str)
            == selected_txid
        ]
    )

    if not selected_graph_nodes.empty:

        st.subheader(
            "Selected Transaction Graph Metrics"
        )

        graph_metric_columns = [
            "node_id",
            "degree",
            "in_degree",
            "out_degree",
            "degree_centrality",
            "connected_component",
        ]

        available_graph_columns = [
            column
            for column in graph_metric_columns
            if column in selected_graph_nodes.columns
        ]

        st.dataframe(
            selected_graph_nodes[
                available_graph_columns
            ],
            width="stretch",
            hide_index=True,
        )


# -------------------------------------------------------------------
# PQC evidence security
# -------------------------------------------------------------------

st.header("PQC Evidence Security")

selected_case = find_case_for_txid(
    selected_txid
)

if selected_case is None:

    st.info(
        "No offline signed case package exists "
        "for the selected transaction."
    )

else:

    evidence_file = (
        selected_case / "evidence.json"
    )

    signature_file = (
        selected_case / "signature.bin"
    )

    report_file = (
        selected_case / "report.json"
    )

    graph_file = (
        selected_case
        / "transaction_graph.json"
    )

    st.write(
        f"**Case:** `{selected_case.name}`"
    )

    try:

        verification = verify_case_package(
            selected_case
        )

        status_col1, status_col2, status_col3 = (
            st.columns(3)
        )

        with status_col1:

            st.metric(
                "Evidence Generated",
                "YES"
                if evidence_file.is_file()
                else "NO",
            )

        with status_col2:

            st.metric(
                "PQC Signature",
                "CREATED"
                if signature_file.is_file()
                else "MISSING",
            )

        with status_col3:

            st.metric(
                "Verification",
                verification["status"],
            )

        st.subheader(
            "Evidence Integrity"
        )

        integrity_col1, integrity_col2 = (
            st.columns(2)
        )

        with integrity_col1:

            if verification[
                "pqc_signature_valid"
            ]:

                st.success(
                    "✓ PQC signature verified"
                )

            else:

                st.error(
                    "✗ PQC signature verification failed"
                )

        with integrity_col2:

            if verification[
                "report_integrity_valid"
            ]:

                st.success(
                    "✓ report.json integrity verified"
                )

            else:

                st.error(
                    "✗ report.json was modified"
                )

        graph_status = (
            verification[
                "transaction_graph_integrity_valid"
            ]
        )

        if graph_status:

            st.success(
                "✓ transaction_graph.json integrity verified"
            )

        else:

            st.error(
                "✗ transaction_graph.json was modified"
            )

        artifact_modified = not (
            verification[
                "report_integrity_valid"
            ]
            and verification[
                "transaction_graph_integrity_valid"
            ]
        )

        st.subheader(
            "Case Security Status"
        )

        if verification["valid"]:

            st.success(
                "Evidence Status: VALID"
            )

            st.success(
                "Modified since signing: NO"
            )

        else:

            st.error(
                "Evidence Status: INVALID"
            )

            if artifact_modified:

                st.error(
                    "Modified since signing: YES"
                )

            else:

                st.warning(
                    "Package verification failed, "
                    "but no report/graph modification was detected."
                )

        st.caption(
            "Verification is performed locally using "
            "the stored case evidence, ML-DSA-65 signature, "
            "and local public key."
        )

    except Exception as exc:

        st.error(
            f"Case verification failed: {exc}"
        )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.divider()

st.caption(
    "Garud-Netra • Offline AI-powered Bitcoin transaction "
    "monitoring and investigative analysis"
)
