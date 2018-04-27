# Sources: "Machine Learning a Probabilistic Perspective, Kevin P. Murphy"

import numpy as np
import math
from scipy.sparse import linalg as sla

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
    
    # Getter and Setter for sigma, update __lambda and __sigma_det every time sigma is set
    @property
    def sigma(self):
        return self.__sigma
    
    @sigma.setter
    def sigma(self, covariance):
        # Check if sigma_det is positive semidefinite if not find the closes positive semidefinite
        try:
            p = (np.linalg.cholesky(covariance))
            self.__sigma = covariance
        except np.linalg.linalg.LinAlgError:
            self.__sigma = self.nearestSPD(covariance)
            
        #print("mtsd sigma: \n{}".format(self.__sigma))
        self.__lambda = np.linalg.inv(np.asarray(self.__sigma))
        #print("mtsd _lambda: \n{}".format(self.__lambda))
        self.__sigma_det = np.linalg.det(self.__sigma)
        
    def pdf(self, data):
        # Murphy, 2.5.3
        # Data is a DxN matrix, where D is dimensionality and N is the number of points to evaluate
        if len(data) is not self.D:
            print("Error: Dimensionality of data and MVST do not agree!")
        else:
            prob = math.exp(self.logpdf(data))
        return prob
    def logpdf(self,data):
        # Data is a DxN matrix, where D is dimensionality and N is number of points to evaluate
        data = np.asarray(data)
        N = np.shape(data)[1]
        logprob = np.zeros((N,1))
        if len(data) is not self.D:
            print("Error: Dimensionality of data and MVST do not agree!")
        else:            
            lognu = math.log(self.nu)
            logdet = math.log(self.__sigma_det)
            b = (self.nu+self.D)/2
            for i in range(0,N):
                    x1 = np.transpose(data[:,i].reshape(self.D,1) - self.mu)
                    x2 = (data[:,i].reshape(self.D,1) - self.mu)
                    mahalanobis = np.matmul(np.matmul(x1,self.__lambda),x2) # Mahalanobis distance is a distance from point to a distribution
                    mahalanobis = mahalanobis[0].real # Mahalanobis has an imaginary part sometimes
                    mahalanobis_imaginary = mahalanobis[0].imag
                    z = math.log1p(mahalanobis/self.nu)
                    logprob[i] = math.lgamma(b) - math.lgamma(self.nu/2) - 0.5*logdet - \
                                 (self.D/2)*(lognu+math.log(math.pi)) - b*z
        return logprob
    # This is needed to handle the error in which the sigma is not positive definite and the nearest positive definite matrix
    # is generated: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
    # nearestSPD() helpers
    def _getAplus(self,A):
        eigval, eigvec = np.linalg.eig(A)
        Q = np.matrix(eigvec)
        xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
        return Q*xdiag*Q.T

    def _getPs(self, A, W=None):
        W05 = np.matrix(W**.5)
        return  W05.I * self._getAplus(W05 * A * W05) * W05.I

    def _getPu(self, A, W=None):
        Aret = np.array(A.copy())
        Aret[W > 0] = np.array(W)[W > 0]
        return np.matrix(Aret)    
    def nearestSPD(self, A, nit=10):
        n = A.shape[0]
        W = np.identity(n) 
        # W is the matrix used for the norm (assumed to be Identity matrix here)
        # the algorithm should work for any diagonal W
        deltaS = 0
        Yk = A.copy()
        for k in range(nit):
            Rk = Yk - deltaS
            Xk = self._getPs(Rk, W=W)
            deltaS = Xk - Rk
            Yk = self._getPu(Xk, W=W)
        return Yk
