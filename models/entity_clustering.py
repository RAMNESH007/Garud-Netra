"""
Entity clustering for the Bitcoin AI Monitoring System.

This module groups transactions/entities according to similar
behavioral feature patterns using DBSCAN.

Cluster label:
    -1 = noise / no dense cluster
     0+ = cluster identifier

Cluster IDs describe behavioral grouping only.
They do not indicate criminality or wrongdoing.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = (
    "data/processed/combined_features.csv"
)

OUTPUT_FILE = (
    "data/processed/cluster_results.csv"
)

MODEL_FILE = (
    "models/dbscan_clusterer.joblib"
)

IDENTIFIER_COLUMNS = [
    "txid",
]


def load_features(input_path):
    """Load the combined feature dataset."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    if df.empty:
        raise ValueError(
            "Combined feature dataset is empty."
        )

    if "txid" not in df.columns:
        raise ValueError(
            "Combined feature dataset must contain txid."
        )

    return df


def select_clustering_features(df):
    """
    Select numeric behavioral features for clustering.

    Identifier columns are excluded.
    """

    candidate_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
    ]

    numeric_columns = (
        df[candidate_columns]
        .select_dtypes(
            include=["number", "bool"]
        )
        .columns
        .tolist()
    )

    if not numeric_columns:
        raise ValueError(
            "No numeric features available for clustering."
        )

    feature_df = df[
        numeric_columns
    ].copy()

    for column in feature_df.columns:

        if feature_df[column].dtype == bool:
            feature_df[column] = (
                feature_df[column]
                .astype(int)
            )

    feature_df = feature_df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    all_missing = [
        column
        for column in feature_df.columns
        if feature_df[column].notna().sum() == 0
    ]

    if all_missing:
        feature_df = feature_df.drop(
            columns=all_missing
        )

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No usable clustering features remain."
        )

    return feature_df


def create_cluster_pipeline():
    """
    Create the offline DBSCAN clustering pipeline.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "dbscan",
                DBSCAN(
                    eps=1.2,
                    min_samples=5,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_entity_clustering(
    input_path,
    output_path=None,
    model_path=None,
):
    """Run DBSCAN clustering."""

    df = load_features(
        input_path
    )

    features = select_clustering_features(
        df
    )

    feature_columns = (
        features.columns.tolist()
    )

    pipeline = create_cluster_pipeline()

    labels = pipeline.fit_predict(
        features
    )

    results = df.copy()

    results["cluster_id"] = labels

    # ---------------------------------------------------------------
    # Cluster statistics
    # ---------------------------------------------------------------

    cluster_sizes = (
        results["cluster_id"]
        .value_counts()
        .sort_index()
    )

    non_noise_labels = [
        label
        for label in cluster_sizes.index
        if label != -1
    ]

    noise_count = int(
        (results["cluster_id"] == -1).sum()
    )

    cluster_count = len(
        non_noise_labels
    )

    # ---------------------------------------------------------------
    # Save results
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
    # Save model metadata
    # ---------------------------------------------------------------

    if model_path is not None:

        model_path = Path(
            model_path
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_package = {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "model_name": "DBSCAN",
            "eps": 1.2,
            "min_samples": 5,
        }

        joblib.dump(
            model_package,
            model_path,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("=" * 60)
    print("ENTITY CLUSTERING")
    print("=" * 60)

    print(
        f"Input rows          : {len(df)}"
    )

    print(
        f"Numeric features    : {len(feature_columns)}"
    )

    print(
        f"Clusters found      : {cluster_count}"
    )

    print(
        f"Noise records       : {noise_count}"
    )

    print("\nFeatures used:")

    for column in feature_columns:
        print(f"✓ {column}")

    print("\nCluster sizes:")

    print(cluster_sizes)

    print("\nValidation:")
    print("✓ Combined features loaded")
    print("✓ txid excluded from clustering")
    print("✓ Numeric behavioral features selected")
    print("✓ Missing values handled")
    print("✓ Features standardized")
    print("✓ DBSCAN clustering completed")
    print("✓ Cluster labels generated")

    if output_path is not None:
        print(
            f"\nSaved results to: {output_path}"
        )

    if model_path is not None:
        print(
            f"Saved model to: {model_path}"
        )

    return results


def main():
    """Run entity clustering."""

    results = run_entity_clustering(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        model_path=MODEL_FILE,
    )

    print("\nFirst 10 records:")

    print(
        results[
            [
                "txid",
                "cluster_id",
            ]
        ].head(10)
    )

    print(
        "\nEntity clustering completed successfully."
    )


if __name__ == "__main__":
    main()
