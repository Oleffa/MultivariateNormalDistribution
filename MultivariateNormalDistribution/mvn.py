# Sources: "Machine Learning a Probabilistic Perspective, Kevin P. Murphy"

import numpy as np
import math
from MultivariateNormalDistribution import mtsd

class MultivariateNormalDistribution:
    
    def __init__(self, dimensions=0, mean=None, S0=None, covariance=None, nu=None, kappa=None, m0=None, name="Unnamed"):
        self.name = name # Name tag for distinguishing different distributions
        self.D = dimensions # Number of dimensions
        self.mu = mean # Mean Vector
        
        self.m0 = m0 # Prior mean for mu
        self.kappa0 = kappa # Weight for m0
        self.nu0 = nu # Weight of S0
        self.S0 = S0 # Prior mean for Sigma
        if covariance is not None: self.sigma = covariance
        
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, covariance):
        self._sigma = covariance
        self._lambda = np.linalg.inv(np.asarray(self._sigma))
        self._sigma_det = np.linalg.det(self._sigma)
        
    def pdf(self):
        # Data is a DxN matrix where D is dimensionality and N is number of points to evaluate
        prob = 0
        if len(data) is not self.D:
            print("Error in logpdf: Dimensionality of data and MVN do not agree!")
        else:
            prob = math.exp(self.logpdf(data)) 
        return 0
    def logpdf(self, data):
        data = np.asarray(data)
        # Data is a DxN matrix, where D is dimensionality and N is number of points to evaluate
        N = np.shape(data)[1]
        
        logprob = np.zeros((N,1))
        if len(data) is not self.D:
            print("Error in logpdf: Dimensionality of data and MVN do not agree!")
        else:
            for i in range(0,N):
                x1 = np.transpose(data[:,i].reshape(self.D,1) - self.mu)
                x2 = (data[:,i].reshape(self.D,1) - self.mu)
                asdf = np.matmul(np.matmul(x1,self._lambda),x2)
                logprob[i] = -0.5*(np.log(self._sigma_det) + self.D*np.log(2*math.pi) + asdf)
        return logprob
    def likelihood(self):
        # Murphy 4.6.3.1
        # p(D|mu,sigma)
        
        # TODO
        return 0
    def loglikelihood(self, data):
        return np.sum(self.logpdf(data))
    def MLE(self, data, update_flag=True):
        # Murphy 4.1.3
        # data is a DxN matrix where D is dimensionality and N is the number of points for MLE
        data = np.asarray(data)
        self.D, N = np.shape(data)
        mu = (np.sum(data,axis=1)/N).reshape(self.D,1)
        incr = np.zeros((self.D, self.D))

        for i in range(0,N):
            column = data[:,i].reshape(self.D,1)
            incr = incr + np.matmul(column,np.transpose(column))

        mu2 = mu*np.transpose(mu)
        sigma = (incr/N)-mu2
        
        # Do mle without writing mu and sigma
        if update_flag is True:
            self.mu = mu
            self.sigma = sigma
    def computePosterior(self,data, update_flag=True):
        # Murphy12, 4.6.3.3
        dim, N = np.shape(data)
        if dim == self.D:
            data_mean = np.average(data,axis=1).reshape(self.D, 1)
            uncentered_scatter_matrix = np.zeros((self.D,self.D))
            for i in range(0,N):
                x = np.matmul(data[:,i].reshape(self.D,1),np.transpose(data[:,i].reshape(self.D,1)))
                uncentered_scatter_matrix = uncentered_scatter_matrix + x # 3x3
            
            # Equation 4.209-214
            kappa_N = self.kappa0 + N # Weight of m0
            m_N = np.divide((self.kappa0*self.m0+N*data_mean),kappa_N)
            nu_N = self.nu0 + N
            S_N = self.S0 + uncentered_scatter_matrix + \
                    self.kappa0*np.matmul(self.m0,np.transpose(self.m0)) -\
                    kappa_N*np.matmul(m_N,np.transpose(m_N))
            # Update prior values
            self.m0 = m_N
            self.kappa0 = kappa_N
            self.S0 = S_N
            self.nu0 = nu_N
            return m_N, kappa_N, nu_N, S_N
        else:
            print("Error: dimensionality is different from MVN dimensionality")
            return
    def MAP(self, data, update_flag=True):
        # Murphy 4.6.3.4
        # MAP estimate (posterior mode)
        m_N, tilde, nu_N, S_N = self.computePosterior(data, update_flag)
        mu = m_N
        sigma = np.divide(S_N,(nu_N + self.D + 2))
        if update_flag is True:
            self.mu = mu
            self.sigma = sigma
        return
    
    def logPosteriorPredictive(self, data, update_flag=True):
        # Murphy 4.6.3.6
        # Uses m0, S0 and so on. Assuming that the computePosterior was overwriting the old prior
        dof = self.nu0 - self.D + 1;
        covariance = (self.kappa0 + 1) * self.S0/(self.kappa0*dof)
        # The posterior predictive given by p(x|D) = p(x,D)/p(D) so it can easily be evaluated in terms
        # of a ratio of marginal likelihoods. It turns out that this ratio has the form of a MVST.
        tst = mtsd.MultivariateTStudentDistribution(self.D, self.m0, covariance, dof)
        predicted_prob = tst.logpdf(data)
        return predicted_prob
    
    def sampleDistribution(self, N):
        # Generate a DxN matrix where D is dimensionality and N is number of samples
        samples = 0
        if N > 0:
            samples = np.random.multivariate_normal(np.squeeze(self.mu), self.sigma, N).T
        return samples
    
