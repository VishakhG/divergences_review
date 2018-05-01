import numpy as np
import numpy.random as npr
import theano as th
import theano.tensor as tt
import theano.sandbox.linalg as tl

# Gaussian mixture model class
# Follows code from
# https://github.com/lucastheis/model-evaluation/tree/master/code/experiments


class GMM():
    def __init__(self, n_gaussians, dim, random_seed):
        self.n_gaussians = n_gaussians
        self.dim = dim
        npr.seed(random_seed)
        self.means, self.covs, self.vars = self.init_gaussian_params()
        self.mix_weights = npr.dirichlet([.5] * n_gaussians)

    def normal(self, X, m, C):
        Z = X - m
        return tt.exp(-tt.sum(Z * tt.dot(tl.matrix_inverse(C), Z), 0) /
                      2. - tt.log(tl.det(C)) / 2. - m.size / 2. *
                      np.log(2. * np.pi))

    def init_gaussian_params(self):
        means = (npr.randn(2, self.n_gaussians) * 1.5)
        vars = 1. / np.square(npr.rand(self.n_gaussians) + 1.)
        covs = ([np.eye(self.dim) * _ for _ in vars])

        return means, covs, vars

    def evaluate_density(self, data):
        data = data.T

        def log_p(X):
            if isinstance(X, tt.TensorVariable):
                return tt.log(tt.sum([self.mix_weights[i] * self.normal(
                    X, self.means[:, [i]], self.covs[i]) for i in
                                      range(len(self.mix_weights))], 0))
            else:
                if log_p.f is None:
                    Y = tt.dmatrix('Y')
                    log_p.f = th.function([Y], log_p(Y))
                return log_p.f(X)
        log_p.f = None

        return log_p(data)

    def get_means(self):
        return self.means.T

    def sample(self, n_samples):

        M = npr.multinomial(n_samples, self.mix_weights)
        data = np.hstack(npr.randn(self.dim, M[i]) * np.sqrt(self.vars[i]) +
                         self.means[:, [i]] for i in range((self.n_gaussians)))
        data = data[:, npr.permutation(n_samples)]

        return data.T
