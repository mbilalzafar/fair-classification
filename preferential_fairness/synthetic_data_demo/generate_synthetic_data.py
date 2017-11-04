import sys
import math
import numpy as np
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data

def generate_synthetic_data(data_type, n_samples):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

    # in the same program run, we might generate this data multiple times (once for single and once for two models)
    # so make sure that the seed is always the same
    SEED = 1234
    seed(SEED)
    np.random.seed(SEED)

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y
    

    if data_type == 1:
    
        disc_factor = math.pi / 8.0 # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination



        """ Generate the non-sensitive features randomly """
        # We will generate one gaussian cluster for each class
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
        mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
        nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
        nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

        # join the posisitve and negative class clusters
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))

        # shuffle the data
        perm = range(0,n_samples*2)
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        
        rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
        X_aux = np.dot(X, rotation_mult)


        """ Generate the sensitive feature here """
        x_control = [] # this array holds the sensitive feature value
        for i in range (0, len(X)):
            x = X_aux[i]

            # probability for each cluster that the point belongs to it
            p1 = nv1.pdf(x)
            p2 = nv2.pdf(x)
            
            # normalize the probabilities from 0 to 1
            s = p1+p2
            p1 = p1/s
            p2 = p2/s
            
            r = np.random.uniform() # generate a random number from 0 to 1

            if r < p1: # the first cluster is the positive class
                x_control.append(1.0) # 1.0 means its male
            else:
                x_control.append(0.0) # 0.0 -> female

        x_control = np.array(x_control)

    elif data_type == 2:

        ## More datasets here.
        pass



    return X,y,x_control

