import numpy as np
from sklearn.datasets import make_blobs
from animator import KMeansAnimator, KMeansInitMethod


if __name__ == "__main__":

        SEED = 0
        
        np.random.seed(SEED)

        X, _ = make_blobs(centers=5, n_samples=1500, random_state = SEED)

        cluster_centers = np.random.randint(X.min(), X.max(), size=(3,2))
        # cluster_centers = None

        params = {
                "n_clusters" : 3,
                "max_iter" : 1,
                "n_init" : 1,
                "max_iter" : 1
        }

        anim = KMeansAnimator(X=X, 
                              k_means_init_method=KMeansInitMethod.CENTROIDS, 
                              k_means_params=params,
                              cluster_centers=cluster_centers)
        
        anim.plot(dir_name="results", file_name="kmeans_random.gif", frames=10, fps=1)

