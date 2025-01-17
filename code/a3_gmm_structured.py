from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

dataDir = "/u/cs401/A3/data/"
#dataDir = "C:\\Users\\LAI\\Desktop\\CSC401\\Speech_Recognition\\data"

class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))
        
    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        this_mu = self.mu[m]
        this_sigma = self.Sigma[m]
        pre_compute1 = np.sum((this_mu ** 2) / this_sigma / 2)
        pre_compute2 = (self._d / 2) * np.log(2 * np.pi)
        pre_compute3 = 0.5 * np.log(np.prod(this_sigma))
        pre_computeM = -(pre_compute1 + pre_compute2 + pre_compute3)
        return pre_computeM

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma
        
def naive_logsumexp(array_like, axis=-1):
    return np.log(np.sum(np.exp(array_like), axis=axis))

def stable_logsumexp(array_like, axis=-1):
    """Compute the stable logsumexp of `vector_like` along `axis`
    This `axis` is used primarily for vectorized calculations.
    """
    array = np.asarray(array_like)
    # keepdims should be True to allow for broadcasting
    m = np.max(array, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(array - m), axis=axis))
        
def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    this_sigma = myTheta.Sigma[m]
    this_mu = myTheta.mu[m]
    term1 = (1 / 2) * (x ** 2) * (np.reciprocal(this_sigma))
    term2 = this_mu * x * (np.reciprocal(this_sigma))
    #Vectorized
    if len(x.shape) > 1:
        return - (np.sum(term1 - term2, axis=1)) + myTheta.precomputedForM(m)
    #Single Row
    else:
        return - (np.sum(term1 - term2, axis=1)) + myTheta.precomputedForM(m)

def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    log_omega = np.log(myTheta.omega)
    log_omegaPlusbs = log_omega + log_Bs
    return log_omega + log_Bs - stable_logsumexp(log_omegaPlusbs, axis=0)

def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    log_omegas = np.log(myTheta.omega)
    log_omegas_plusBs = log_omegas + log_Bs
    return np.sum(stable_logsumexp(log_omegas_plusBs, axis=0))


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    T = X.shape[0]
    d = X.shape[1]
    i = 0
    prev_L = -float('inf')
    improvement = float('inf')
    # for ex.,
    myTheta.reset_omega(np.full((M,1), 1/M))
    random_indices = random.sample(range(T), M)
    myTheta.reset_mu(X[random_indices])
    myTheta.reset_Sigma(np.full((M,d), 1.0))

    while i <= maxIter and improvement >= epsilon:
        
        #Compute Intermediate Results
        log_bs = np.zeros((M, T))
        for m in range(M):
            log_bs[m] = log_b_m_x(m, X, myTheta)
        log_ps = log_p_m_x(log_bs, myTheta)
        log_Lik = logLik(log_bs, myTheta)
            
        ps = np.exp(log_ps)
        summation_ps = np.sum(ps, axis=1)
        ps_Dot_X = np.dot(ps, X)
        ps_Dot_XSuare = np.dot(ps, np.square(X))
        #Update parameters
        for m in range(M):
            myTheta.omega[m] = summation_ps[m] / T
            myTheta.mu[m] = ps_Dot_X[m] / summation_ps[m]
            myTheta.Sigma[m] = (ps_Dot_XSuare[m] / summation_ps[m] ) - np.square(myTheta.mu[m])
        
        improvement = log_Lik - prev_L
        prev_L = log_Lik
        i += 1
                
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    bestLikelihood = -float('inf')
    models_likelihood = {}
    T, d = mfcc.shape
    M = models[0].omega.shape[0]
    
    for i in range(len(models)):
        cur_myTheta = models[i]
        log_bs = np.zeros((M, T))
        for m in range(M):
            log_bs[m] = log_b_m_x(m, mfcc, cur_myTheta)
        log_Lik = logLik(log_bs, cur_myTheta)
        models_likelihood[i] = log_Lik
        if (log_Lik > bestLikelihood):
            bestLikelihood = log_Lik
            bestModel = i
        
    #Sorted by Best Likelihood in models_likelihood
    sorted_models_likelihood = sorted(models_likelihood.items() ,reverse=True, key=lambda x: x[1])
    
    if k > 0:
        f = open("gmmLiks.txt","a")
        f.write('{}\n'.format(models[correctID].name))
        for j in range(k):
            modelID = sorted_models_likelihood[j][0]
            likelihood = sorted_models_likelihood[j][1]
            f.write('{} {}\n'.format(models[modelID].name, likelihood))
        f.close()
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    #Experiemt
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print("Accuracy: {0: 1.4f} \n".format(accuracy))
