import pandas as pd

from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features

from src.clustering.kmeans import run_kmeans
from src.clustering.hierarchical import run_hierarchical
from src.clustering.dbscan import run_dbscan
from src.clustering.gmm import run_gmm

from src.evaluation import evaluate_clustering
from src.utils import save_csv, print_metrics, cluster_summary

# Load dataset
df = pd.read_csv("customer-segmentation-unsupervised-saikumar\Data\Raw\used_cars_data.csv")

# Step 1: Preprocess
df_clean = preprocess_data(df)

# Step 2: Feature Engineering
features = create_features(df_clean)

# Step 3: Clustering Models
kmeans_labels = run_kmeans(features)
hier_labels = run_hierarchical(features)
dbscan_labels = run_dbscan(features)
gmm_labels = run_gmm(features)

# Step 4: Evaluation
kmeans_metrics = evaluate_clustering(features, kmeans_labels)
hier_metrics = evaluate_clustering(features, hier_labels)
dbscan_metrics = evaluate_clustering(features, dbscan_labels)
gmm_metrics = evaluate_clustering(features, gmm_labels)

# Step 5: Print Metrics
print_metrics("KMeans", kmeans_metrics)
print_metrics("Hierarchical", hier_metrics)
print_metrics("DBSCAN", dbscan_metrics)
print_metrics("GMM", gmm_metrics)

# Step 6: Save Results
df_clean['KMeans'] = kmeans_labels
df_clean['Hierarchical'] = hier_labels
df_clean['DBSCAN'] = dbscan_labels
df_clean['GMM'] = gmm_labels

save_csv(df_clean, "results/clustered_data.csv")

# Step 7: Cluster Summary
summary = cluster_summary(df_clean, 'KMeans')
print("\nCluster Summary:\n", summary)