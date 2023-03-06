import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from animator import Animator
from sklearn import datasets


np.random.seed()

if __name__ == "__main__":
        X, _ = make_blobs(centers=8, n_samples=1500,random_state = 1)

        # iris = datasets.load_iris()
        # X = iris.data[:, :2]  # we only take the first two features.

        #specify centroid starting position, uncomment line below
        n1 = 0
        n2 = 2

        params = {
                "n_clusters" : 16,
                "max_iter" : 1,
                "n_init"     : 1,
                "max_iter"   : 1,
                # "init" : np.array([X[n1,:], X[n2,:]]) #or "random" / "kmeans++"
                "init" : "random"
        }

        Animator(X=X, algorithm=KMeans,params=params, save = True, name="example.gif").plot()

