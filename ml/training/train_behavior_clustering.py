import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "clustering_features.csv"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "clustering_metadata.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "results"
)

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

K_VALUES = range(2, 9)

# Use a sample for expensive validation metrics.
# The full dataset will still be used for final training.
SAMPLE_SIZE = 100_000


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING CLUSTERING DATASET")
    print("=" * 70)

    print(f"\nInput:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nDataset shape: {df.shape}")

    if "session_id" in df.columns:
        session_ids = df["session_id"].copy()
        X = df.drop(columns=["session_id"])
    else:
        session_ids = None
        X = df.copy()

    print(f"Feature matrix: {X.shape}")

    return df, session_ids, X


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(X):

    print("\n" + "=" * 70)
    print("VALIDATING CLUSTERING DATA")
    print("=" * 70)

    missing = X.isna().sum().sum()

    print(f"\nMissing values: {missing}")

    if missing > 0:
        raise ValueError("Missing values detected.")

    infinite_count = np.isinf(X.to_numpy()).sum()

    print(f"Infinite values: {infinite_count}")

    if infinite_count > 0:
        raise ValueError("Infinite values detected.")

    print("\nData validation passed.")


# ============================================================
# SAMPLE DATA FOR METRICS
# ============================================================

def create_sample(X):

    if len(X) <= SAMPLE_SIZE:
        return X

    print(
        f"\nUsing {SAMPLE_SIZE:,} samples "
        f"for expensive cluster validation metrics."
    )

    return X.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


# ============================================================
# CLUSTER EVALUATION
# ============================================================

def evaluate_clusters(X):

    print("\n" + "=" * 70)
    print("EVALUATING CLUSTER COUNTS")
    print("=" * 70)

    results = []

    X_sample = create_sample(X)

    for k in K_VALUES:

        print("\n" + "-" * 70)
        print(f"Testing K = {k}")
        print("-" * 70)

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(X_sample)

        silhouette = silhouette_score(
            X_sample,
            labels
        )

        davies_bouldin = davies_bouldin_score(
            X_sample,
            labels
        )

        calinski = calinski_harabasz_score(
            X_sample,
            labels
        )

        inertia = model.inertia_

        print(f"Silhouette Score       : {silhouette:.6f}")
        print(f"Davies-Bouldin Score   : {davies_bouldin:.6f}")
        print(f"Calinski-Harabasz      : {calinski:.2f}")
        print(f"Inertia                : {inertia:.2f}")

        results.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "calinski_harabasz_score": calinski,
                "inertia": inertia,
            }
        )

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# SELECT BEST K
# ============================================================

def select_best_k(results_df):

    print("\n" + "=" * 70)
    print("SELECTING BEST CLUSTER COUNT")
    print("=" * 70)

    # Primary criterion:
    # highest silhouette score.

    best_row = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    best_k = int(best_row["k"])

    print(
        f"\nBest K according to silhouette score: {best_k}"
    )

    print("\nCandidate results:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return best_k


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(X, best_k):

    print("\n" + "=" * 70)
    print("TRAINING FINAL K-MEANS MODEL")
    print("=" * 70)

    print(f"\nClusters: {best_k}")
    print(f"Training samples: {len(X):,}")

    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(X)

    print("\nFinal clustering complete.")

    return model, labels


# ============================================================
# CLUSTER SUMMARY
# ============================================================

def create_cluster_summary(df, X, labels):

    print("\n" + "=" * 70)
    print("CREATING CLUSTER SUMMARY")
    print("=" * 70)

    result = df.copy()

    result["cluster"] = labels

    summary = []

    for cluster_id in sorted(result["cluster"].unique()):

        cluster_data = result[
            result["cluster"] == cluster_id
        ]

        row = {
            "cluster": int(cluster_id),
            "sessions": len(cluster_data),
            "percentage": (
                len(cluster_data)
                / len(result)
                * 100
            ),
        }

        for feature in X.columns:

            if feature in cluster_data.columns:

                row[f"{feature}_mean"] = (
                    cluster_data[feature].mean()
                )

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    print("\nCluster sizes:")

    print(
        summary_df[
            [
                "cluster",
                "sessions",
                "percentage",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    return result, summary_df


# ============================================================
# PCA VISUALIZATION DATA
# ============================================================

def create_pca_data(X, labels):

    print("\n" + "=" * 70)
    print("CREATING PCA REPRESENTATION")
    print("=" * 70)

    # PCA is used for visualization only.
    # It does NOT replace the original clustering features.

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "cluster": labels,
        }
    )

    explained = pca.explained_variance_ratio_

    print(
        f"\nPC1 explained variance: "
        f"{explained[0]:.4f}"
    )

    print(
        f"PC2 explained variance: "
        f"{explained[1]:.4f}"
    )

    print(
        f"Total explained variance: "
        f"{explained.sum():.4f}"
    )

    return pca_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE CYBERSECURITY DECEPTION")
    print("BEHAVIOURAL CLUSTERING")
    print("=" * 70)

    # Load
    df, session_ids, X = load_dataset()

    # Validate
    validate_data(X)

    # Evaluate K
    results_df = evaluate_clusters(X)

    # Save evaluation results
    evaluation_file = (
        OUTPUT_DIR / "cluster_evaluation.csv"
    )

    results_df.to_csv(
        evaluation_file,
        index=False
    )

    print(
        f"\nEvaluation results saved to:\n"
        f"{evaluation_file}"
    )

    # Select K
    best_k = select_best_k(results_df)

    # Final model
    model, labels = train_final_model(
        X,
        best_k
    )

    # Cluster summary
    clustered_df, summary_df = create_cluster_summary(
        df,
        X,
        labels
    )

    # Save clustered dataset
    clustered_file = (
        OUTPUT_DIR / "clustered_sessions.csv"
    )

    clustered_df.to_csv(
        clustered_file,
        index=False
    )

    # Save summary
    summary_file = (
        OUTPUT_DIR / "cluster_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # PCA
    pca_df = create_pca_data(
        X,
        labels
    )

    pca_file = (
        OUTPUT_DIR / "pca_clusters.csv"
    )

    pca_df.to_csv(
        pca_file,
        index=False
    )

    # Save metadata
    metadata = {
        "algorithm": "KMeans",
        "random_state": RANDOM_STATE,
        "best_k": best_k,
        "total_sessions": len(X),
        "feature_count": X.shape[1],
        "features": list(X.columns),
    }

    metadata_output = (
        OUTPUT_DIR / "cluster_model_metadata.json"
    )

    with open(
        metadata_output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)

    print(f"\nBest K: {best_k}")

    print("\nGenerated files:")

    print(
        f"1. {evaluation_file}"
    )

    print(
        f"2. {clustered_file}"
    )

    print(
        f"3. {summary_file}"
    )

    print(
        f"4. {pca_file}"
    )

    print(
        f"5. {metadata_output}"
    )

    print("\nIMPORTANT:")
    print(
        "Do not assign attacker meanings to clusters yet."
    )
    print(
        "We will interpret the clusters in the next research stage."
    )


if __name__ == "__main__":
    main()