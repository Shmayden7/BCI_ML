import numpy as np

class LDA:

    def __init__(self, nComponents):
        self.nComponents = nComponents
        self.linearDiscriminant = None

    def fit(self, X, y):
        nFeatures = X.shape[1]
        classLabels = np.unique(y)

        # S_W (within class), S_B (between class)
        meanOverall = np.mean(X, axis=0)

        #initialize 0 arrays with dimensions nFeatures x nFeatures
        S_W = np.zeros((nFeatures, nFeatures))
        S_B = np.zeros((nFeatures, nFeatures))

        for x in classLabels:
            
