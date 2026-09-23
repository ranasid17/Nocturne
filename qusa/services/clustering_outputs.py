"""Detailed regime statistics and noninteractive, run-scoped figures."""

import pandas as pd


def cluster_statistics(data):
    rows = []
    for label, group in data.groupby("cluster"):
        def column(name):
            return group[name] if name in group else pd.Series(dtype=float)

        rows.append({
            "cluster": int(label), "size": len(group), "percent": len(group) / len(data) * 100,
            "overnight_delta_mean": column("overnight_delta_pct").mean(),
            "overnight_delta_median": column("overnight_delta_pct").median(),
            "overnight_delta_std": column("overnight_delta_pct").std(),
            "volume_mean": column("volume_ratio").mean(),
            "volume_ratio_mean": column("volume_ratio").mean(),
            "volume_spikes": column("volume_spike").sum() if "volume_spike" in group else None,
            "rsi_mean": column("rsi").mean(),
            "rsi_oversold": (column("rsi") < 30).sum() if "rsi" in group else None,
            "rsi_overbought": (column("rsi") > 70).sum() if "rsi" in group else None,
            "abnormal_rate": column("abnormal").mean() * 100,
        })
    return rows


def save_cluster_figures(data, analyzer, optimal, directory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Clustering plots require nocturne-qusa[research].") from exc

    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(figure, name):
        path = directory / f"{name}.png"
        try:
            figure.tight_layout()
            figure.savefig(path, dpi=120)
            paths.append(path)
        finally:
            plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(optimal["k"], optimal["inertia"], marker="o")
    axes[0].set(xlabel="Cluster count", ylabel="Inertia")
    axes[1].plot(optimal["k"], optimal["silhouette_score"], marker="o")
    axes[1].set(xlabel="Cluster count", ylabel="Silhouette score")
    save(figure, "elbow_curve")

    components, pca = analyzer.perform_pca(data)
    figure, axis = plt.subplots(figsize=(8, 5))
    points = axis.scatter(components[:, 0], components[:, 1], c=analyzer.cluster_labels, cmap="tab10", s=15)
    axis.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
             ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    figure.colorbar(points, ax=axis, label="Cluster")
    save(figure, "pca_clusters")

    profiles = data.loc[data["cluster"] >= 0].groupby("cluster")[analyzer.feature_columns].mean()
    normalized = (profiles - profiles.mean()) / profiles.std().replace(0, 1)
    figure, axis = plt.subplots(figsize=(10, 5))
    heatmap = axis.imshow(normalized.fillna(0), cmap="RdBu_r", aspect="auto")
    axis.set_xticks(range(len(profiles.columns)), labels=profiles.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(profiles)), labels=profiles.index)
    axis.set_ylabel("Cluster")
    figure.colorbar(heatmap, ax=axis, label="Standardized mean")
    save(figure, "cluster_profiles")

    figure, axis = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(data["date"]) if "date" in data else data.index
    y = data["close"] if "close" in data else data["overnight_delta_pct"]
    axis.plot(x, y, color="gray", linewidth=0.6)
    axis.scatter(x, y, c=data["cluster"], cmap="tab10", s=10)
    axis.set(xlabel="Session", ylabel="Close" if "close" in data else "Overnight change (%)")
    save(figure, "cluster_time_series")
    return paths
