from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum
from sklearn.cluster import KMeans


class KMeansInitMethod(Enum):
    RANDOM = "random"
    KMEANSPLUSPLUS = "k-means++"
    CENTROIDS = "centroids"


class KMeansAnimator():
    
    def __init__(self, 
                 X: np.ndarray, 
                 k_means_init_method: KMeansInitMethod, 
                 k_means_params: Dict,
                 cluster_centers: Optional[np.ndarray] = None):
        
        self.fig, self.ax = plt.subplots()
        self.params = k_means_params

        self.dataset = X

        if k_means_init_method == KMeansInitMethod.CENTROIDS:

            if cluster_centers is None:
                raise ValueError("Centroids must be provided when using KMeansInitMethod.CENTROIDS")

            self.cluster_centers_ = cluster_centers
            self.params["init"] = self.cluster_centers_
        
        else:
            self.params["init"] = k_means_init_method.value

            temp_algo = KMeans(**self.params).fit(self.dataset)
            self.cluster_centers_ = temp_algo.cluster_centers_

        self.scatter_points = self.ax.scatter(self.dataset[:,0], self.dataset[:,1])
        self.scatter_cluster_centers_ = self.ax.scatter(self.cluster_centers_[:,0], self.cluster_centers_[:,1])
        
    def init(self):
        self.scatter_points.set_offsets(self.dataset)
        self.scatter_cluster_centers_.set_offsets(self.cluster_centers_)

        return self.scatter_points, self.scatter_cluster_centers_,


    def update(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        self.scatter_points.set_offsets(self.dataset)

        # set colors only after second iteration so that no clusters are shown at the beginning
        if frame != 0: 
            self.scatter_points.set_array(self.clustering)
        
        self.scatter_cluster_centers_.set_offsets(self.cluster_centers_)
        self.scatter_cluster_centers_.set_color([(1.0,0.0,0.0) for _ in range(self.cluster_centers_.shape[0])])

        self.params["init"] = self.cluster_centers_
        fitted_model = KMeans(**self.params).fit(self.dataset)
    
        self.cluster_centers_ = fitted_model.cluster_centers_
        self.clustering = fitted_model.labels_

        return self.scatter_points, self.scatter_cluster_centers_,

    def plot(self, 
             fps: int, 
             frames: int,
             save: bool = False,
             dir_name: Optional[str] = None, 
             file_name: Optional[str] = None):

        anim = FuncAnimation(self.fig, 
                             self.update, 
                             frames = frames,
                             interval = 1000, 
                             repeat = False,
                             blit=True, 
                             init_func=self.init)
        
        if save:

            if dir_name is None:
                raise ValueError("dir_name must be provided when saving the animation")
            if file_name is None:
                raise ValueError("file_name must be provided when saving the animation")

            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

            anim.save(os.path.join(dir_name, file_name), writer=PillowWriter(fps=fps))
        else:

            plt.show()

        





        