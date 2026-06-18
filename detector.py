import numpy as np

class AnomalyDetector:
    def __init__(self, thresh=3.0): self.thresh = thresh
    def fit(self, d): self.mean, self.std = d.mean(0), d.std(0)
    def predict(self, d):
        z = np.abs((d-self.mean)/(self.std+1e-8))
        return z.mean(1) > self.thresh
