"""
Transaction graph construction for the Bitcoin AI Monitoring System.

The graph represents:

    Bitcoin Address -> Transaction -> Bitcoin Address

NetworkX is used to construct and analyze the graph.

The module calculates:
- node degree
- degree centrality
- connected components
- transaction relationships
"""

from pathlib import Path

import networkx as nx
import pandas as pd


INPUT_FILE = "data/processed/correlated_data.csv"

OUTPUT_NODES = "data/processed/graph_nodes.csv"
OUTPUT_EDGES = "data/processed/graph_edges.csv"


def split_values(value):
    """Convert an address field into a list of values."""

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    for separator in ["|", ";"]:
        value = value.replace(separator, ",")

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def build_transaction_graph(df):
    """
    Build a directed transaction graph.

    Nodes:
        - address
        - transaction

    Edges:
        address -> transaction
        transaction -> address
    """

    required_columns = [
        "txid",
        "input_addresses",
        "output_addresses",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    graph = nx.DiGraph()

    for _, row in df.iterrows():

        txid = str(row["txid"]).strip()

        if not txid:
            continue

        transaction_node = f"tx:{txid}"

        graph.add_node(
            transaction_node,
            node_type="transaction",
            txid=txid,
        )

        input_addresses = split_values(
            row["input_addresses"]
        )

        output_addresses = split_values(
            row["output_addresses"]
        )

        # -----------------------------------------------------------
        # Input addresses -> transaction
        # -----------------------------------------------------------

        for address in input_addresses:

            address_node = f"addr:{address}"

            graph.add_node(
                address_node,
                node_type="address",
                address=address,
            )

            graph.add_edge(
                address_node,
                transaction_node,
                relationship="input",
            )

        # -----------------------------------------------------------
        # Transaction -> output addresses
        # -----------------------------------------------------------

        for address in output_addresses:

            address_node = f"addr:{address}"

            graph.add_node(
                address_node,
                node_type="address",
                address=address,
            )

            graph.add_edge(
                transaction_node,
                address_node,
                relationship="output",
            )

    return graph


def calculate_graph_features(graph):
    """Calculate explainable graph features for every node."""

    degree = dict(graph.degree())

    in_degree = dict(
        graph.in_degree()
    )

    out_degree = dict(
        graph.out_degree()
    )

    centrality = nx.degree_centrality(
        graph
    )

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    component_lookup = {}

    for component_id, component in enumerate(
        components,
        start=1,
    ):

        for node in component:
            component_lookup[node] = component_id

    rows = []

    for node, attributes in graph.nodes(
        data=True
    ):

        rows.append(
            {
                "node_id": node,
                "node_type": attributes.get(
                    "node_type"
                ),
                "address": attributes.get(
                    "address"
                ),
                "txid": attributes.get(
                    "txid"
                ),
                "degree": degree.get(
                    node,
                    0,
                ),
                "in_degree": in_degree.get(
                    node,
                    0,
                ),
                "out_degree": out_degree.get(
                    node,
                    0,
                ),
                "degree_centrality": centrality.get(
                    node,
                    0.0,
                ),
                "connected_component": component_lookup.get(
                    node
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_edge_features(graph):
    """Convert graph edges into a tabular representation."""

    rows = []

    for source, target, attributes in graph.edges(
        data=True
    ):

        rows.append(
            {
                "source": source,
                "target": target,
                "relationship": attributes.get(
                    "relationship"
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    """Build and analyze the transaction graph."""

    input_path = Path(INPUT_FILE)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    graph = build_transaction_graph(
        df
    )

    node_features = calculate_graph_features(
        graph
    )

    edge_features = calculate_edge_features(
        graph
    )

    Path(OUTPUT_NODES).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_features.to_csv(
        OUTPUT_NODES,
        index=False,
    )

    edge_features.to_csv(
        OUTPUT_EDGES,
        index=False,
    )

    address_nodes = sum(
        1
        for _, attributes
        in graph.nodes(data=True)
        if attributes.get("node_type")
        == "address"
    )

    transaction_nodes = sum(
        1
        for _, attributes
        in graph.nodes(data=True)
        if attributes.get("node_type")
        == "transaction"
    )

    print("=" * 60)
    print("TRANSACTION GRAPH CONSTRUCTION")
    print("=" * 60)

    print(
        f"Input transactions : {len(df)}"
    )

    print(
        f"Graph nodes         : {graph.number_of_nodes()}"
    )

    print(
        f"Graph edges         : {graph.number_of_edges()}"
    )

    print(
        f"Address nodes       : {address_nodes}"
    )

    print(
        f"Transaction nodes   : {transaction_nodes}"
    )

    print(
        f"Connected components: "
        f"{nx.number_weakly_connected_components(graph)}"
    )

    print("\nValidation:")
    print("✓ Transaction records loaded")
    print("✓ Address fields parsed")
    print("✓ Transaction nodes created")
    print("✓ Address nodes created")
    print("✓ Input relationships created")
    print("✓ Output relationships created")
    print("✓ Degree calculated")
    print("✓ Centrality calculated")
    print("✓ Connected components calculated")

    print(
        f"\nSaved node features to: {OUTPUT_NODES}"
    )

    print(
        f"Saved edge features to: {OUTPUT_EDGES}"
    )

    print(
        "\nTransaction graph construction completed successfully."
    )


if __name__ == "__main__":
    main()
