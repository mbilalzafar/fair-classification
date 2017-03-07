from __future__ import division
import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data
from sklearn.linear_model import LogisticRegression
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)
sys.path.insert(0, '../../fair_classification/') 
import utils as ut


def generate_synthetic_data(data_type, plot_data=False):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        Non sensitive features will be drawn from a 2D gaussian distribution.
        Sensitive feature specifies the demographic group of the data point and can take values 0 and 1.

        The code will generate data such that a classifier optimizing for accuracy will lead to disparate misclassification rates for the two demographic groups.
        You can generate different data configurations using different values for the "data_type" parameter.
    """

    n_samples = 1000 # generate these many data points per cluster

    def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
        """
        mean_in: mean of the gaussian cluster
        cov_in: covariance matrix
        z_val: sensitive feature value
        class_label: +1 or -1
        n: number of points
        """

        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n)
        y = np.ones(n, dtype=float) * class_label
        z = np.ones(n, dtype=float) * z_val # all the points in this cluster get this value of the sensitive attribute

        return nv, X, y, z


    if data_type == 1:

        """
        Generate data such that a classifier optimizing for accuracy will have disparate false positive rates as well as disparate false negative rates for both groups.
        """


        cc = [[10,1], [1,4]]
        mu1, sigma1 = [2, 3], cc  # z=1, +
        cc = [[5,2], [2,5]]
        mu2, sigma2 = [1, 2], cc  # z=0, +

        cc = [[5, 1], [1, 5]]
        mu3, sigma3 = [-5,0], cc # z=1, -
        cc = [[7, 1], [1, 7]]
        mu4, sigma4 = [0,-1], cc # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, +1, int(n_samples * 1) ) # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1) ) # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, -1, int(n_samples * 1) ) # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, -1, int(n_samples * 1) ) # z=0, -

    elif data_type == 2:

        """
        Generate data such that a classifier optimizing for accuracy will have disparate false positive rates for both groups but will have equal false negative rates.
        """


        cc = [[3,1], [1,3]]
        mu1, sigma1 = [2, 2], cc  # z=1, +
        mu2, sigma2 = [2, 2], cc  # z=0, +

        mu3, sigma3 = [-2,-2], cc # z=1, -
        cc = [[3,3], [1,3]]
        mu4, sigma4 = [-1,0], cc # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, +1, int(n_samples * 1) ) # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1) ) # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, -1, int(n_samples * 1) ) # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, -1, int(n_samples * 1) ) # z=0, -



    # merge the clusters
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    x_control = np.hstack((z1, z2, z3, z4))

    # shuffle the data
    perm = range(len(X))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    x_control = x_control[perm]

    
    """ Plot the data """
    if plot_data:
        plt.figure()
        num_to_draw = 200 # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]

        plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=60, linewidth=2, label= "group-0 +ve")
        plt.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='red', marker='x', s=60, linewidth=2, label = "group-0 -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=60, linewidth=2, label = "group-1 +ve")
        plt.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='red', marker='o', facecolors='none', s=60, linewidth=2, label = "group-1 -ve")


        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=21)
        plt.ylim((-8,12))

        plt.savefig("img/data.png")
        plt.show()


    x_control = {"s1": x_control} # all the sensitive features are stored in a dictionary
    X = ut.add_intercept(X)
    

    return X,y,x_control

