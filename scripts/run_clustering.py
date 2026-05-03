# qusa/scripts/run_clustering.py

import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.analysis.clustering import ClusterAnalyzer
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QUSA clustering analysis for one ticker."
    )
    parser.add_argument("ticker", help="Ticker symbol to cluster, for example AMZN")
    return parser.parse_args()


def confirm_directory(path):
    """
    Confirm that a directory exists, creating it if necessary.

    Parameters:
        1) path (str): The directory path to confirm.
    """

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return


def plot_elbow_curve(optimal_results, paths, logger):
    """
    Plots the elbow curve for clustering analysis.

    Parameters:
        1) optimal_results (dict): Dictionary containing the
            optimal number of clusters and associated metrics.
        2) paths (dict): Dictionary containing paths for saving figures.
        3) logger (logging.Logger): Logger object for logging messages.
    """

    logger.info("Generating elbow curve plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # inertia plot
    ax1.plot(
        optimal_results["k"], optimal_results["inertia"], marker="o", color="#3A445D"
    )
    ax1.axvline(
        x=optimal_results["optimal_k"],
        color="#274C77",
        linestyle="--",
        label=f"Optimal Clusters: {optimal_results['optimal_k']}",
    )
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Method for Optimal k", fontsize=14, fontweight="bold")
    ax1.legend()

    # silhouette score plot
    ax2.plot(
        optimal_results["k"],
        optimal_results["silhouette_score"],
        marker="o",
        color="#3A445D",
    )
    ax2.axvline(
        x=optimal_results["optimal_k"],
        color="#274C77",
        linestyle="--",
        label=f"Optimal Clusters: {optimal_results['optimal_k']}",
    )
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Scores for Different k", fontsize=14, fontweight="bold")
    ax2.legend()

    plt.tight_layout()

    fig_path = os.path.join(paths["figures_dir"], "elbow_curve.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Elbow curve saved → {fig_path}")

    plt.show()

    return


def plot_pca_clusters(data, pca_X, pca_model, paths, logger):
    """
    Plot clusters in PC space.

    Parameters:
        1) data (pd.DataFrame): The original dataset with cluster labels.
        2) pca_X (np.ndarray): The PCA-transformed data.
        3) pca_model (PCA): The fitted PCA model.
        4) paths (dict): Dictionary containing paths for saving figures.
        5) logger (logging.Logger): Logger object for logging messages.
    """

    logger.info("Generating PCA cluster plot...")

    # copy data for plotting
    data_filtered = data.loc[data["cluster"] != -1].copy()

    # plotting code
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        pca_X[:, 0],
        pca_X[:, 1],
        c=data_filtered["cluster"].values,
        cmap="tab10",
        alpha=0.7,
    )
    cbar = plt.colorbar(scatter, ax=ax)  # add colorbar
    cbar.set_label("Cluster Label", fontsize=12)

    var1 = pca_model.explained_variance_ratio_[0] * 100  # percentage variance for PC1
    var2 = pca_model.explained_variance_ratio_[1] * 100  # percentage variance for PC2

    ax.set_xlabel(f"Principal Component 1 ({var1:.2f}% Variance)", fontsize=12)
    ax.set_ylabel(f"Principal Component 2 ({var2:.2f}% Variance)", fontsize=12)
    ax.set_title("PCA Clustering Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(paths["figures_dir"], "pca_clusters.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ PCA cluster plot saved → {fig_path}")

    plt.show()

    return


def plot_cluster_profiles(analyzer, paths, logger):
    """
    Plot heatmaps of cluster profiles.

    Parameters:
        1) analyzer (ClusterAnalyzer): ClusterAnalyzer instance
            with clustering results.
        2) paths (dict): Dictionary containing paths for saving figures.
        3) logger (logging.Logger): Logger object for logging messages.
    """

    logger.info("Generating cluster profiles heatmap...")

    profiles = analyzer.cluster_profiles

    # select feature columns
    feature_cols = [
        col for col in profiles.columns if col not in ["cluster", "count", "percent"]
    ]

    # prepare data for heatmap
    heatmap_data = profiles[feature_cols].T
    heatmap_data.columns = [f"Cluster {int(col)}" for col in profiles["cluster"]]

    # clean feature names
    heatmap_data.index = [
        col.replace("_mean", "").replace("_", " ").title() for col in heatmap_data.index
    ]

    # plotting code
    fig_height = max(6, 0.5 * len(heatmap_data))  # dynamic figure height by map size
    fig, ax = plt.subplots(figsize=(12, fig_height))
    cax = ax.matshow(heatmap_data, cmap="viridis", aspect="auto")
    fig.colorbar(cax)
    ax.set_xlabel("Clusters", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title("Cluster Profiles Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(paths["figures_dir"], "cluster_profiles_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Cluster profile heatmap saved → {fig_path}")

    plt.show()

    return


def plot_cluster_time_series(data, paths, logger):
    """
    Plot time series of cluster distributions.

    Parameters:
        1) data (pd.DataFrame): The original dataset with cluster labels and timestamps.
        2) paths (dict): Dictionary containing paths for saving figures.
        3) logger (logging.Logger): Logger object for logging messages.
    """

    logger.info("Generating cluster time series plots...")

    # copy data for plotting
    data_filtered = data.loc[data["cluster"] != -1].copy()

    # convert timestamp to datetime if not already
    data_filtered["date"] = pd.to_datetime(data_filtered["date"])

    # plotting code
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ## plot 1: overnight delta vs cluster
    scatter = ax1.scatter(
        data_filtered["date"],
        data_filtered["overnight_delta"],
        c=data_filtered["cluster"],
        cmap="tab10",
        alpha=0.7,
    )
    cbar1 = plt.colorbar(scatter, ax=ax1)  # add colorbar for clusters
    cbar1.set_label("Cluster Label", fontsize=12)

    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    ax1.set_xlabel("date", fontsize=12)
    ax1.set_ylabel("Overnight Delta", fontsize=12)
    ax1.set_title("Overnight Delta vs Time by Cluster", fontsize=14, fontweight="bold")

    ## plot 2: cluster distribution over time
    data_filtered["month"] = data_filtered["date"].dt.to_period("M")  # label months
    cluster_counts = (
        data_filtered.groupby(["month", "cluster"])  # group by month and cluster
        .size()
        .unstack(fill_value=0)
    )
    cluster_props = cluster_counts.div(  # normalize to get proportions
        cluster_counts.sum(axis=1), axis=0
    )

    cluster_props.plot(kind="area", stacked=True, ax=ax2, colormap="tab10", alpha=0.7)
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Number of Days", fontsize=12)
    ax2.set_title("Cluster Distribution Over Time", fontsize=14, fontweight="bold")
    ax2.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    fig_path = os.path.join(paths["figures_dir"], "cluster_time_series.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Cluster time series plot saved → {fig_path}")

    plt.show()

    return


def analyze_clusters(data, analyzer, logger):
    """
    Analyze and log characteristics of each cluster.

    Parameters:
        1) data (pd.DataFrame): Dataset with cluster labels.
        2) analyzer (ClusterAnalyzer): Fitted clustering analyzer.
        3) logger (logging.Logger): Logger instance.
    """

    logger.info("=" * 80)
    logger.info("DETAILED CLUSTER ANALYSIS")
    logger.info("=" * 80)

    interpretations = analyzer.interpret_clusters(data)

    for cluster_label, interpretation in interpretations.items():
        cluster_data = data.loc[data["cluster"] == cluster_label]

        if cluster_data.empty:
            logger.debug(f"Cluster {cluster_label} is empty — skipping.")
            continue

        size = len(cluster_data)
        pct = size / len(data) * 100

        logger.info("-" * 80)
        logger.info(f"CLUSTER {cluster_label}: {interpretation}")
        logger.info("-" * 80)
        logger.info(f"Size: {size} days ({pct:.1f}%)")

        # Overnight delta
        logger.info("Overnight Delta:")
        logger.info(
            f"  Mean: {cluster_data['overnight_delta_pct'].mean():.2f}% | "
            f"Median: {cluster_data['overnight_delta_pct'].median():.2f}% | "
            f"Std: {cluster_data['overnight_delta_pct'].std():.2f}%"
        )

        # Volume
        logger.info("Volume:")
        logger.info(
            f"  Mean Ratio: {cluster_data['volume_ratio'].mean():.2f}x | "
            f"Spikes (>2x): {(cluster_data['volume_spike']).sum()} days"
        )

        # RSI
        logger.info("RSI:")
        logger.info(
            f"  Mean: {cluster_data['rsi'].mean():.1f} | "
            f"Oversold (<30): {(cluster_data['rsi'] < 30).sum()} | "
            f"Overbought (>70): {(cluster_data['rsi'] > 70).sum()}"
        )

        # Abnormal moves
        if "abnormal" in cluster_data.columns:
            abnormal_rate = cluster_data["abnormal"].mean() * 100
            logger.info(f"Abnormal Overnight Moves: {abnormal_rate:.1f}%")

        # Day-of-week effects
        if "day_of_week" in cluster_data.columns:
            logger.info("Day of Week Distribution:")
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

            dow_dist = (
                cluster_data["day_of_week"].value_counts(normalize=True).sort_index()
                * 100
            )

            for dow, pct in dow_dist.items():
                if dow in dow_map:
                    logger.info(f"  {dow_map[dow]}: {pct:.1f}%")

    logger.info("=" * 80)
    logger.info("END OF CLUSTER ANALYSIS")
    logger.info("=" * 80)

    return


def export_cluster_statistics(data, analyzer, paths, logger):
    """
    Export cluster statistics to a CSV file.

    Parameters:
        1) data (pd.DataFrame): Dataset with cluster labels.
        2) analyzer (ClusterAnalyzer): Fitted clustering analyzer.
        3) paths (dict): Dictionary containing paths for saving files.
        4) logger (logging.Logger): Logger instance.
    """

    logger.info("Exporting cluster statistics to CSV...")

    if hasattr(analyzer, "cluster_profiles"):
        cluster_profiles = analyzer.cluster_profiles.copy()
    else:
        cluster_profiles = pd.DataFrame()

    # aggregate additional stats
    cluster_statistics = []

    for cluster_label in sorted(data["cluster"].unique()):
        cluster_data = data.loc[data["cluster"] == cluster_label]

        if cluster_data.empty:
            logger.debug(f"Cluster {cluster_label} is empty — skipping.")
            continue

        stats = {
            "cluster": cluster_label,
            "size": len(cluster_data),
            "percent": len(cluster_data) / len(data) * 100,
            "overnight_delta_mean": cluster_data.get(
                "overnight_delta_pct", pd.Series()
            ).mean(),
            "overnight_delta_median": cluster_data.get(
                "overnight_delta_pct", pd.Series()
            ).median(),
            "overnight_delta_std": cluster_data.get(
                "overnight_delta_pct", pd.Series()
            ).std(),
            "volume_mean": cluster_data.get("volume_ratio", pd.Series()).mean(),
            "volume_spikes": cluster_data.get("volume_spike", pd.Series()).sum(),
            "volume_ratio_mean": cluster_data["volume_ratio"].mean(),
            "rsi_mean": cluster_data.get("rsi", pd.Series()).mean(),
            "rsi_oversold": (cluster_data.get("rsi", pd.Series()) < 30).sum(),
            "rsi_overbought": (cluster_data.get("rsi", pd.Series()) > 70).sum(),
        }

        if "abnormal" in cluster_data.columns:
            stats["abnormal_rate"] = cluster_data["abnormal"].mean() * 100

        cluster_statistics.append(stats)

    df_cluster_stats = pd.DataFrame(cluster_statistics)

    # Save JSON
    json_path = os.path.join(paths["processed_data_dir"], "cluster_statistics.json")
    df_cluster_stats.to_json(json_path, orient="records", indent=4)
    logger.info(f"✓ Cluster stats exported to JSON → {json_path}")

    return


def main():
    """
    Main function to run clustering analysis and visualizations.
    """

    args = parse_args()
    ticker = args.ticker.upper()

    logger = setup_logger(
        "ClusteringPipeline",
        log_file=str(PROJECT_ROOT / "logs" / "clustering.log"),
    )

    logger.info("=" * 80)
    logger.info("Starting Clustering Analysis Pipeline")
    logger.info("=" * 80)

    # load config
    try:
        logger.info("Loading configuration...")
        config = load_config(PROJECT_ROOT / "qusa" / "utils" / "config.yaml")
        data_cfg = config["data"]
        paths = data_cfg["paths"]
        logger.info("✓ Configuration loaded")

    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return 1

    try:
        logger.info("Loading processed data...")
        processed_dir = paths["processed_data_dir"]

        data_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        data = pd.read_csv(data_path)
        logger.info(f"✓ Data loaded: {data.shape}")

    except Exception as e:
        logger.error(f"✗ Failed to load processed data: {e}")
        return 1

    confirm_directory(os.path.join(paths["figures_dir"], "dummy.txt"))

    try:
        logger.info("Running clustering analysis...")
        analyzer = ClusterAnalyzer(n_clusters=4, algorithm="kmeans")

        optimal_results = analyzer.find_optimal_clusters(data, max_k=8)
        logger.info(f"✓ Optimal k = {optimal_results['optimal_k']}")

        # plot cluster elbow curve
        plot_elbow_curve(optimal_results, paths, logger)

        data_clustered = analyzer.fit_clusters(data, feature_cols=None)
        logger.info("✓ Clustering complete")

        # apply PCA for visualization
        pca_X, pca_model = analyzer.perform_pca(
            data_clustered,
            feature_cols=analyzer.feature_columns,
        )
        logger.info(
            f"✓ PCA explained variance: {pca_model.explained_variance_ratio_.sum():.1%}"
        )
        # visualizations
        plot_pca_clusters(data_clustered, pca_X, pca_model, paths, logger)
        plot_cluster_profiles(analyzer, paths, logger)
        plot_cluster_time_series(data_clustered, paths, logger)

        try:  # detailed cluster analysis
            analyze_clusters(data=data_clustered, analyzer=analyzer, logger=logger)
        except Exception as e:
            logger.error(f"Cluster analysis failed: {e}")
            logger.exception("Full traceback:")

    except Exception as e:
        logger.exception(f"✗ Error during clustering analysis: {e}")
        return 1

    try:  # save cluster statistics
        export_cluster_statistics(data_clustered, analyzer, paths, logger)
    except Exception as e:
        logger.error(f"Failed to export cluster stats: {e}")
        logger.exception("Full traceback:")

    try:  # save clustered data
        logger.info("Saving clustered data...")
        output_path = os.path.join(
            paths["processed_data_dir"],
            f"{ticker}_processed_clustered.csv",
        )
        data_clustered.to_csv(output_path, index=False)
        logger.info(f"✓ Clustered data saved → {output_path}")

    except Exception as e:
        logger.error(f"✗ Failed to save clustered data: {e}")
        return 1

    logger.info("=" * 80)
    logger.info("✓ CLUSTERING ANALYSIS COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
