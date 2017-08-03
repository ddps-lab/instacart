from __future__ import division
import numpy as np

class MatrixFactorization:
    def __init__(self, rating, dimension, steps=200, gamma=0.01, lmd=0.05, threshold=0.001):
        #rating data
        self.rating = rating
        #average rating for all users and all items
        self.mu = np.mean(rating[rating > 0])
        #bias for user
        self.b_u = self.b_u()
        #bias for item
        self.b_i = self.b_i()
        #p represent users latent factor
        self.p = np.random.rand(rating.shape[0], dimension)/rating.shape[0]
        #q represent items latent factor
        self.q = np.random.rand(dimension, rating.shape[1])/rating.shape[1]
        #learning patramaters
        self.gamma = gamma
        self.lmd = lmd
        self.steps = steps
        self.threshold = threshold

    def b_u(self):
        """
        function for set b_u(bias for user)
        @return array of each users bias
        """
        ret = np.mean(self.rating, axis=1, keepdims=True)
        items = self.rating.shape[1]
        for row in xrange(ret.shape[0]):
            nonzero = np.count_nonzero(self.rating[row])
            if nonzero != 0:
                ret[row] *= items / nonzero
        return ret - self.mu

    def b_i(self):
        """
        function for set b_i(bias for item)
        @return array of each items bias
        """
        ret = np.mean(self.rating, axis=0)
        users = self.rating.shape[0]
        for col in xrange(ret.shape[0]):
            nonzero = np.count_nonzero(self.rating[:, col])
            if nonzero != 0:
                ret[col] *= users / nonzero
        return ret - self.mu

    def get_bias(self):
        """
        function to get user-item bias matrix(mu + b_u + b_i)
        @return user-item bias matrix
        """
        return np.zeros((self.rating.shape[0], self.rating.shape[1])) + self.mu + self.b_u + self.b_i

    def get_error(self):
        """
        function to get regularized error value(object of minimization)
        @return regularized error value
        """
        rating_error = np.sum((self.rating - np.dot(self.p, self.q) - self.get_bias()) ** 2)
        regularize = np.linalg.norm(self.p) ** 2 + np.linalg.norm(self.q) ** 2 + np.linalg.norm(self.b_u) ** 2 + np.linalg.norm(self.b_i) ** 2
        return rating_error + self.lmd * regularize

    def update(self):
        """
        function to update using stochastic gradient descent(SGD)
        """
        error = self.rating - np.dot(self.p, self.q) - self.get_bias()
        #remove unrated user-item pair from update
        error[np.where(self.rating == 0)] = 0
        temp_q = self.q.copy()
        self.q += self.gamma * (np.dot(self.p.T, error) - self.lmd * self.q)
        self.p += self.gamma * (np.dot(error, temp_q.T) - self.lmd * self.p)
        self.b_u += self.gamma * (np.mean(error, axis=1, keepdims=True) - self.lmd * self.b_u)
        self.b_i += self.gamma * (np.mean(error, axis=0) - self.lmd * self.b_i)

    def run(self):
        """
        function to run learning
        @return predicted rating matrix
        """
        for i in xrange(self.steps):
            self.update()
            error = self.get_error()
            if(error < self.threshold):
                break
        ret = np.dot(self.p, self.q) + self.get_bias()
        return ret

if __name__ == '__main__':
    R = np.array([[5, 3, 0, 1],[4, 0, 0, 1],[1, 3, 0, 0],[1, 0, 0, 4],[0, 1, 5, 4]])
    mf = MatrixFactorization(rating=R, dimension=2)
    nR = mf.run()
    print R
<<<<<<< HEAD
    print nR
=======
    print nR
>>>>>>> 4e79622a392568e52e2d6c02f6299959adfdd719
