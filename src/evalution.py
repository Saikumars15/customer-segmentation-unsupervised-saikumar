from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    results = {}

    # Silhouette Score
    if len(set(labels)) > 1:
        results['silhouette_score'] = silhouette_score(X, labels)
    else:
        results['silhouette_score'] = -1

    # Davies-Bouldin Index
    results['davies_bouldin_score'] = davies_bouldin_score(X, labels)

    # Calinski-Harabasz Score
    results['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)

    return results