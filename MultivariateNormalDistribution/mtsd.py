import numpy as np

class MultivariateTStudentDistribution:
    def __init__(self, dimensions, mean, covariance, dof):
        # No Prior Knowledge
        self.D = dimensions
        self.mu = mean
        self.sigma = covariance
        self.nu = dof
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, covariance):
        self._sigma = covariance
        self._lambda = np.linalg.inv(np.asarray(self._sigma))
        self._sigma_det = np.linalg.det(self._sigma)
    def pdf(self, data):
        return
    def logpdf(self,data):
        return
    def laplaceApproximation(self):
        return
