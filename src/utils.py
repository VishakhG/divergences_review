import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def ravel(params):
    """
    Combine parameters into a long one-dimensional array.
    @type  params: C{list}
    @param params: list of shared variables
    @rtype: C{ndarray}
    """
    return np.hstack(p.get_value().ravel() for p in params)


def plot_density_data(pdf, data, xmin=-5, xmax=7, ymin=-5, ymax=7):

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 200),
        np.linspace(ymin, ymax, 200))

    grid = np.asarray([xx.ravel(), yy.ravel()]).T

    zz = np.exp(pdf(grid))
    zz = zz.reshape(xx.shape)
    hh, x, y = np.histogram2d(data[:, 0], data[:, 1], 80,
                              range=[(xmin, xmax), (ymin, ymax)])

    sns.set_style('whitegrid')
    sns.set_style('ticks')
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(hh.T[::-1], extent=[x[0], x[-1], y[0], y[-1]],
               interpolation='nearest', cmap='YlGnBu_r')
    plt.contour(xx, yy, zz, 7, colors='w', alpha=.7)
    plt.axis('equal')
    plt.axis([x[0], x[-1], y[0], y[-1]])
    plt.axis('off')
    plt.gcf().tight_layout()


def pdf_normal(mu, cov):
    def pdf(X):
        var = multivariate_normal(mean=mu, cov=cov)
        return var.pdf(X)

    return pdf


class Generator(nn.Module):
    def __init__(self, data):
        super(Generator, self).__init__()
        mu = np.mean(data, 0)
        std = np.std(data - mu)
        self.mu = nn.Parameter(torch.Tensor(mu))

        self.std = nn.Parameter(torch.Tensor(std))

    def get_params(self):
        return self.mu, self.std

    def forward(self, z):

        output = self.mu * z + self.std
        return output
