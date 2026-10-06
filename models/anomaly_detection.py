"""
Anomaly detection for the Bitcoin AI Monitoring System.

This module:
- Loads combined engineered features
- Selects numeric ML features
- Handles missing values
- Trains an Isolation Forest model
- Produces anomaly scores
- Produces binary anomaly labels
- Saves anomaly results
- Saves the trained pipeline for later offline inference

Anomaly label:
    0 = normal
    1 = suspicious/anomalous

The anomaly score is a relative 0-100 ranking within the
current dataset. It is not a probability.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


INPUT_FILE = (
    "data/processed/combined_features.csv"
)

OUTPUT_FILE = (
    "data/processed/anomaly_results.csv"
)

MODEL_FILE = (
    "models/isolation_forest.joblib"
)

IDENTIFIER_COLUMNS = [
    "txid",
]


def load_features(input_path):
    """Load the combined feature dataset."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input feature file not found: {input_path}"
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


def select_numeric_features(df):
    """
    Select numeric columns for anomaly detection.

    Identifier columns such as txid are deliberately excluded.
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
            "No numeric features available for anomaly detection."
        )

    numeric_df = df[
        numeric_columns
    ].copy()

    # Convert boolean values to integers.
    for column in numeric_df.columns:
        if numeric_df[column].dtype == bool:
            numeric_df[column] = (
                numeric_df[column]
                .astype(int)
            )

    # Replace invalid infinite values.
    numeric_df = numeric_df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # Check that every feature has at least one usable value.
    all_missing = [
        column
        for column in numeric_df.columns
        if numeric_df[column].notna().sum() == 0
    ]

    if all_missing:
        numeric_df = numeric_df.drop(
            columns=all_missing
        )

    if numeric_df.shape[1] == 0:
        raise ValueError(
            "All numeric features contain only missing values."
        )

    return numeric_df


def create_model():
    """
    Create the offline anomaly-detection pipeline.

    Median imputation handles missing numeric values.
    Isolation Forest performs unsupervised anomaly detection.
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


def calculate_relative_scores(decision_values):
    """
    Convert Isolation Forest decision values into relative
    0-100 anomaly scores.

    Isolation Forest gives lower decision values to more
    anomalous observations, therefore we invert the values
    before ranking them.
    """

    raw_anomaly_values = -decision_values

    scores = (
        pd.Series(raw_anomaly_values)
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    return scores.round(2)


def run_anomaly_detection(
    input_path,
    output_path=None,
    model_path=None,
):
    """
    Train Isolation Forest and generate anomaly results.

    Returns
    -------
    pandas.DataFrame
        Original feature rows plus anomaly outputs.
    """

    df = load_features(
        input_path
    )

    numeric_features = select_numeric_features(
        df
    )

    feature_columns = (
        numeric_features.columns.tolist()
    )

    model = create_model()

    # ---------------------------------------------------------------
    # Train model
    # ---------------------------------------------------------------

    model.fit(
        numeric_features
    )

    # ---------------------------------------------------------------
    # Predict anomaly labels
    # ---------------------------------------------------------------

    predictions = model.predict(
        numeric_features
    )

    # Isolation Forest:
    #     1  = inlier
    #    -1  = outlier
    #
    # Convert this to the project's convention:
    #     0  = normal
    #     1  = suspicious

    anomaly_labels = (
        predictions == -1
    ).astype(int)

    # ---------------------------------------------------------------
    # Calculate anomaly scores
    # ---------------------------------------------------------------

    decision_values = (
        model.decision_function(
            numeric_features
        )
    )

    anomaly_scores = calculate_relative_scores(
        decision_values
    )

    # ---------------------------------------------------------------
    # Create result dataset
    # ---------------------------------------------------------------

    results = df.copy()

    results["anomaly_score"] = (
        anomaly_scores.values
    )

    results["anomaly_label"] = (
        anomaly_labels
    )

    results["model_decision_value"] = (
        decision_values.round(6)
    )

    # Sort most anomalous records first.
    results = results.sort_values(
        by="anomaly_score",
        ascending=False,
    ).reset_index(
        drop=True
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
    # Save model + feature metadata
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
            "pipeline": model,
            "feature_columns": feature_columns,
            "model_name": "IsolationForest",
            "random_state": 42,
        }

        joblib.dump(
            model_package,
            model_path,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    anomaly_count = int(
        results["anomaly_label"].sum()
    )

    normal_count = (
        len(results) - anomaly_count
    )

    print("=" * 60)
    print("ANOMALY DETECTION")
    print("=" * 60)

    print(
        f"Input rows          : {len(df)}"
    )

    print(
        f"Numeric features    : {len(feature_columns)}"
    )

    print(
        f"Normal records      : {normal_count}"
    )

    print(
        f"Anomalous records   : {anomaly_count}"
    )

    print(
        f"Minimum score       : "
        f"{results['anomaly_score'].min():.2f}"
    )

    print(
        f"Maximum score       : "
        f"{results['anomaly_score'].max():.2f}"
    )

    print("\nFeatures used by model:")

    for column in feature_columns:
        print(f"✓ {column}")

    print("\nValidation:")
    print("✓ Combined features loaded")
    print("✓ txid excluded from ML features")
    print("✓ Numeric features selected")
    print("✓ Missing values handled")
    print("✓ Isolation Forest trained")
    print("✓ Anomaly labels generated")
    print("✓ Relative anomaly scores generated")

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
    """Run anomaly detection."""

    results = run_anomaly_detection(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        model_path=MODEL_FILE,
    )

    print("\nTop 10 anomalous records:")

    print(
        results[
            [
                "txid",
                "anomaly_score",
                "anomaly_label",
            ]
        ].head(10)
    )

    print(
        "\nAnomaly detection completed successfully."
    )


if __name__ == "__main__":
    main()
