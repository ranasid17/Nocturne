# QUSA/qusa/analysis/clustering.py

import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Performs clustering analysis on financial time series data.
    """

    def __init__(self, n_clusters=4, algorithm="kmeans", random_state=42):
        """
        Class constructor.

        Parameters:
            1) n_clusters (int): Number of clusters for clustering algorithms that require it.
            2) algorithm (str): Clustering algorithm to use ('kmeans' or 'dbscan').
            3) random_state (int): Random state for reproducibility.
        """

        self.cluster_labels = None
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state

        # model objects initialized after fitting model
        self.scaler = None
        self.pca = None
        self.model = None

        # feature information
        self.feature_columns = None
        self.cluster_profiles = None

    def prepare_features(self, df, feature_cols=None):
        """
        Prepare features for clustering.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.
            2) feature_cols (list): List of column names to use as features.

        Returns;
            1) tuple: (scaled_features, feature_names, valid_indices)
        """

        # handle case where no feature columns are provided
        if feature_cols is None or len(feature_cols) == 0:
            feature_cols = [
                "overnight_delta_pct",
                "intraday_return",
                "volume_ratio",
                "rsi",
                "atr_pct",
                "close_position",
                "52_week_high_proximity",
                "52_week_low_proximity",
            ]

        # filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]

        # handle case of no available indicator columns
        if len(available_cols) == 0:
            raise ValueError("No valid feature columns found in DataFrame.")

        # store feature names as attribute
        self.feature_columns = available_cols

        # remove rows with NaNs in feature columns
        df_features = df[available_cols].dropna().reset_index()

        # check if enough data points remain after NaN removal
        if len(df_features) < self.n_clusters:
            raise ValueError(
                f"Not enough valid samples ({len(df_features)})"
                f"for {self.n_clusters} clusters."
            )

        # handle case where no scaler exists yet
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(df_features[available_cols])
        # otherwise use existing scaler
        else:
            X_scaled = self.scaler.transform(df_features[available_cols])

        return X_scaled, available_cols, df_features["index"].values

    def find_optimal_clusters(self, df, feature_cols=None, max_k=10):
        """
        Find the optimal number of clusters using
        elbow method and silhouette score.

        Parameters:
            1) X (np.ndarray): Scaled feature array.
            3) feature_cols (list, opt): List of features for clustering.
            3) max_k (int): Maximum number of clusters to test.


        Returns:
            1) best_k (int): Optimal number of clusters.
        """

        # call method to prepare features
        X, _, _ = self.prepare_features(df, feature_cols)

        results = {"k": [], "inertia": [], "silhouette_score": []}

        # iterate across available cluster counts (k)
        for k in range(2, max_k + 1):
            # perform k-means clustering with current k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)

            # store metrics from current k
            results["k"].append(k)
            results["inertia"].append(kmeans.inertia_)
            results["silhouette_score"].append(score)

        # apply elbow method
        inertia = np.array(results["inertia"])
        delta = np.diff(inertia)
        delta_ratio = delta[:-1] / delta[1:]
        elbow_k = np.argmax(delta_ratio) + 2  # +2 to adjust for k=2 start

        results["optimal_k"] = elbow_k

        return results

    def fit_clusters(self, df, feature_cols):
        """
        Fit clustering model to the data and add labels to clusters.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.
            2) feature_cols (list): List of column names to use as features.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with an additional 'Cluster_Label' column.
        """

        # prepare features for clustering
        X_scaled, feature_names, valid_indices = self.prepare_features(df, feature_cols)

        # fit clustering model based on selected algorithm
        if self.algorithm == "kmeans":
            model = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
        elif self.algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

        # fit model and predict cluster labels
        cluster_labels = model.fit_predict(X_scaled)

        # store fitted objects and labels as attributes
        self.model = model
        self.cluster_labels = cluster_labels

        # create modified DataFrame with cluster labels
        df_mod = df.copy()
        df_mod["cluster"] = -1  # default label for NaN rows
        df_mod.loc[valid_indices, "cluster"] = cluster_labels

        self._calculate_cluster_profiles(df_mod)

        return df_mod

    def _calculate_cluster_profiles(self, df):
        """
        Calculate mean feature values for each cluster.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Cluster_Label' column.
        """

        # filter rows with valid cluster labels
        df_valid = df[df["cluster"] != -1]

        # calculate mean feature values per cluster
        cluster_profiles = df_valid.groupby("cluster")[self.feature_columns].mean()

        profiles = []
        for cluster_label in sorted(df_valid["cluster"].unique()):
            # filter data to current cluster
            cluster_data = df_valid[df_valid["cluster"] == cluster_label]

            # store current cluster element count and proportion
            profile = {
                "cluster": cluster_label,
                "count": len(cluster_data),
                "proportion": len(cluster_data) / len(df_valid),
            }

            # store mean element count per feature for current cluster
            for feature in self.feature_columns:
                if feature in cluster_profiles.columns:
                    profile[f"mean_{feature}"] = cluster_profiles.loc[
                        cluster_label, feature
                    ]

            profiles.append(profile)

        # store cluster profiles as attribute
        self.cluster_profiles = pd.DataFrame(profiles)

        return

    def perform_pca(self, df, feature_cols=None, n_components=2):
        """
        Apply PCA to reduce feature dimensions.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.
            2) feature_cols (list): List of feature column names.
            3) n_components (int): Number of PCA components to keep.

        Returns:
            1) df_mod (pd.DataFrame): DataFrame with PCA components added.
        """

        # prepare features for PCA
        X_scaled, _, _ = self.prepare_features(df, self.feature_columns)

        # fit PCA model
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)

        return X_pca, self.pca

    def get_cluster_summary(self):
        """
        Get the cluster profiles summary.

        Returns:
            1) cluster_profiles (pd.DataFrame): DataFrame summarizing cluster profiles.
        """

        # handle case where cluster profiles have not been calculated
        if self.cluster_profiles is None:
            raise ValueError("Cluster profiles have not been calculated yet.")

        return self.cluster_profiles

    def interpret_clusters(self, df):
        """
        Interpret clusters by characteristics.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Cluster_Label' column.

        Returns:
            1) interpretations (dict): Dictionary interpreting each cluster.
        """

        # handle case where cluster profiles have not been calculated
        if self.cluster_profiles is None:
            self._calculate_cluster_profiles(df)

        interpretations = {}

        for _, row in self.cluster_profiles.iterrows():
            cluster_label = int(row["cluster"])

            # extract key characteristics for current cluster
            overnight = row.get("mean_overnight_delta_pct", 0)
            volume = row.get("mean_volume_ratio", 0)
            rsi = row.get("mean_rsi", 0)

            # interpret with logic rules using feature thresholds
            if (volume > 1.0) and (abs(overnight) > 2.0):
                interpretation = "High Volume Spike with Significant Overnight Change"
            elif (abs(overnight) < 0.5) and (volume < 1.2):
                interpretation = "Low Volatility, Stable Trading Day"
            elif (overnight > 1.0) and (rsi > 70):
                interpretation = "Momentum Up"
            elif (overnight < -1.0) and (rsi < 30):
                interpretation = "Momentum Down"
            elif rsi > 70:
                interpretation = "Overbought Conditions"
            elif rsi < 30:
                interpretation = "Oversold Conditions"
            else:
                interpretation = "Standard Trading Day"

            interpretations[cluster_label] = interpretation

        return interpretations

    def extract_feature_importance(self):
        """
        Calculate feature importance by cluster separation.

        Returns:
            1) feature_importance (pd.DataFrame): DataFrame ranking features by importance.
        """

        # handle case where cluster profiles have not been calculated
        if self.cluster_labels is None:
            raise ValueError("Must call fit_clusters first")

        importance = []

        for col in self.feature_columns:
            mean_col = f"mean_{col}"

            if mean_col in self.cluster_profiles.columns:
                # calculate variance of mean feature values across clusters
                values = self.cluster_profiles[mean_col].values
                variance = np.var(values)

                # add epsilon to avoid division by zero
                scale = np.mean(np.abs(values)) + 1e-10

                importance.append(
                    {
                        "feature": col,
                        "variance_across_clusters": variance,
                        "importance": variance / (np.mean(np.abs(values)) + 1e-10),
                    }
                )

        if not importance:
            raise ValueError("No valid features found for importance calculation.")

        # convert list to DataFrame and sort by importance
        df_importance = pd.DataFrame(importance)
        df_importance = df_importance.sort_values(
            by="importance", ascending=False
        ).reset_index(drop=True)

        return df_importance

    def predict_cluster(self, new_data):
        """
        Predict cluster label for new data points.

        Parameters:
            1) new_data (pd.DataFrame): DataFrame containing new stock data.

        Returns:
            1) new_cluster_labels (np.ndarray): Predicted cluster labels for new data.
        """

        # handle case where model has not been fitted
        if self.model is None or self.scaler is None:
            raise ValueError("Clustering model has not been fitted yet.")

        # extract and scale features from new data
        X_new_scaled, _, _ = self.prepare_features(new_data, self.feature_columns)

        # predict cluster labels using fitted model
        new_cluster_labels = self.model.predict(X_new_scaled)

        return new_cluster_labels

    def print_summary(self, df):
        """
        Print a summary of clustering results.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Cluster_Label' column.
        """

        # handle case where cluster profiles have not been calculated
        if self.cluster_profiles is None:
            self._calculate_cluster_profiles(df)

        logger.info("\n" + "=" * 80)
        logger.info("CLUSTERING ANALYSIS SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nAlgorithm: {self.algorithm}")
        logger.info(f"Number of clusters: {self.n_clusters}")
        logger.info(f"Features used: {len(self.feature_columns)}")
        logger.info(f"Total observations: {len(df[df['cluster'] >= 0])}")

        # cluster interpretations
        interpretations = self.interpret_clusters(df)

        logger.info("\n" + "-" * 80)
        logger.info("CLUSTER PROFILES")
        logger.info("-" * 80)

        for _, row in self.cluster_profiles.iterrows():
            cluster_label = int(row["cluster"])
            label = interpretations.get(cluster_label, "Unknown")

            logger.info(f'\nCluster {cluster_label}: "{label}"')
            logger.info(f"  Size: {row['count']} days ({row['proportion']*100:.1f}%)")
            logger.info(
                f"  Avg overnight delta: {row.get('mean_overnight_delta_pct', 0):.2f}%"
            )
            logger.info(f"  Avg volume ratio: {row.get('mean_volume_ratio', 1):.2f}x")
            logger.info(f"  Avg RSI: {row.get('mean_rsi', 50):.1f}")

        # Feature importance
        logger.info("\n" + "-" * 80)
        logger.info("FEATURE IMPORTANCE (Top 5)")
        logger.info("-" * 80)

        importance_df = self.extract_feature_importance()
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        # PCA explained variance
        if self.pca is not None:
            logger.info("\n" + "-" * 80)
            logger.info("PCA COMPONENTS")
            logger.info("-" * 80)

            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            logger.info(f"  PC1 explains: {explained_variance[0]:.1%} of variance")
            logger.info(f"  PC2 explains: {explained_variance[1]:.1%} of variance")
            logger.info(f"  Total explained: {cumulative_variance[1]:.1%}")

        logger.info("\n" + "=" * 80)

        return
