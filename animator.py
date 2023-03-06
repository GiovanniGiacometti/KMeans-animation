from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import os
import numpy as np

class Animator():
    
    def __init__(self, X, algorithm,params,save,name,fps=1,frames=10):
        
        self.fig, self.ax = plt.subplots()
        self.algorithm = algorithm
        self.params = params
        self.fps = fps
        self.frames = frames

        self.save = save

        self.dataset = X

        if type(self.params["init"])==np.ndarray:
            self.cluster_centers_ = self.params["init"]
        else:
            fittedmodel = self.algorithm(**self.params).fit(self.dataset)
            self.cluster_centers_ = fittedmodel.cluster_centers_

        self.scatter_points = self.ax.scatter(self.dataset[:,0], self.dataset[:,1])
        self.scatter_cluster_centers_ = self.ax.scatter(self.cluster_centers_[:,0], self.cluster_centers_[:,1])
        self.name = name
        
    def init(self):
        self.scatter_points.set_offsets(self.dataset)
        self.scatter_cluster_centers_.set_offsets(self.cluster_centers_)

        return self.scatter_points, self.scatter_cluster_centers_,


    def update(self,frame):
        self.scatter_points.set_offsets(self.dataset)

        if frame != 0: #set colors only after second iteration so that no clusters are shown at the beginning
            self.scatter_points.set_array(self.clustering)
        
        self.scatter_cluster_centers_.set_offsets(self.cluster_centers_)
        self.scatter_cluster_centers_.set_color([(1.0,0.0,0.0) for _ in range(self.cluster_centers_.shape[0])])

        self.params["init"] = self.cluster_centers_
        fitted_model = self.algorithm(**self.params).fit(self.dataset)
    
        self.cluster_centers_ = fitted_model.cluster_centers_
        self.clustering = fitted_model.labels_

        return self.scatter_points, self.scatter_cluster_centers_,

    
    def plot(self):
        anim = FuncAnimation(self.fig, self.update, frames = self.frames ,interval = 1000, repeat = False ,blit=True, init_func=self.init)
        

        if self.save:
            dir_name = "results"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            anim.save(f"{dir_name}/{self.name}", writer=PillowWriter(fps=self.fps) )
        else:
            plt.show()

        





        