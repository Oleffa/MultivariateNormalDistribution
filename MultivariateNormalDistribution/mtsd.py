# Sources: "Machine Learning a Probabilistic Perspective, Kevin P. Murphy"

import numpy as np
import math

class MultivariateTStudentDistribution:
    # The Multivariate Student t distribution is a more robust alternative to the MVN. It is used in
    # mvn.py to calculate the posterior predictive for the MAP parameter estimation.
    # The posterior predictive given by p(x|D) = p(x,D)/p(D) so it can easily be evaluated in terms
    # of a ratio of marginal likelihoods. It turns out that this ratio has the form of a MVST.
    
    
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
        # Murphy, 2.5.3
        # Data is a DxN matrix, where D is dimensionality and N is the number of points to evaluate
        if len(data) is not self.D:
            print("Error: Dimensionality of data and MVST do not agree!")
        else:
            prob = math.exp(self.logpdf(data))
        return prob
    def logpdf(self,data):
        data = np.asarray(data)
        # Data is a DxN matrix, where D is dimensionality and N is number of points to evaluate
        if len(data) is not self.D:
            print("Error: Dimensionality of data and MVST do not agree!")
        else:
            N = np.shape(data)[1]
            logprob = np.zeros((N,1))
            
            lognu = math.log(self.nu)
            logdet = math.log(self._sigma_det)
            b = (self.nu+self.D)/2
            for i in range(0,N):
                    x1 = np.transpose(data[:,i].reshape(self.D,1) - self.mu)
                    x2 = (data[:,i].reshape(self.D,1) - self.mu)
                    mahalanobis = np.matmul(np.matmul(x1,self._lambda),x2) # Mahalanobis distance is a distance from point to a distribution
                    z = math.log1p(mahalanobis/self.nu)
                    logprob[i] = math.lgamma(b) - math.lgamma(self.nu/2) - 0.5*logdet - \
                                 (self.D/2)*(lognu+math.log(math.pi)) - b*z
        return logprob
    def laplaceApproximation(self):
        # Bishop 4.4
        # TODO
        # Needed?
        return
