import os
import pandas as pd
import matplotlib.pyplot as plt

# Create directory if not exists
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Save dataframe
def save_csv(df, path):
    df.to_csv(path, index=False)

# Save plot
def save_plot(fig, path):
    fig.savefig(path)
    plt.close(fig)

# Generate cluster summary
def cluster_summary(df, cluster_col):
    summary = df.groupby(cluster_col).mean()
    return summary

# Print nicely
def print_metrics(name, metrics):
    print(f"\n{name} Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")