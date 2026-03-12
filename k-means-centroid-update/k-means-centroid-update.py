import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points = np.array(points)
    assignments = np.array(assignments)

    centroids = []

    for i in range(k):
        cluster_points = points[assignments == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid.tolist())

    return centroids