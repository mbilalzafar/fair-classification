from __future__ import division
import os,sys
import numpy as np
from generate_synthetic_data import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # for plotting stuff
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
from plot_synthetic_boundaries import plot_data

import stats_pref_fairness as compute_stats # for computing stats
from linear_clf_pref_fairness import LinearClf


def print_stats_and_plots(x,y,x_sensitive, clf, fname):

    dist_arr, dist_dict = clf.get_distance_boundary(x, x_sensitive)
    acc, _, acc_stats = compute_stats.get_clf_stats(dist_arr, dist_dict, y, x_sensitive, print_stats=True)

    if isinstance(clf.w, dict):
        w_arr = [clf.w[0], clf.w[1]]
        lt_arr = ['c--', 'b--']
        label_arr = [
                        "$\mathcal{B}_0: %0.2f; \mathcal{B}_1: %0.2f$" % (acc_stats[0][0]["frac_pos"], acc_stats[1][0]["frac_pos"]),
                        "$\mathcal{B}_0: %0.2f; \mathcal{B}_1: %0.2f$" % (acc_stats[0][1]["frac_pos"], acc_stats[1][1]["frac_pos"]),
                    ]
    else:
        w_arr = [clf.w]
        lt_arr = ['k--']
        label_arr = [
                "$\mathcal{B}_0: %0.2f; \mathcal{B}_1: %0.2f$" % (acc_stats[0][0]["frac_pos"], acc_stats[1][1]["frac_pos"]),
            ]

    title = "$Acc: %0.2f$" % acc
    plot_data(x, y, x_sensitive, w_arr, label_arr, lt_arr, fname, title)




def test_synthetic_data():
    
    """ Generate the synthetic data """
    data_type = 1
    X, y, x_sensitive = generate_synthetic_data(data_type=data_type, n_samples=1000) # set plot_data to False to skip the data plot
    X = compute_stats.add_intercept(X)

    """ Split the data into train and test """
    TEST_FOLD_SIZE = 0.3
    x_train, x_test, y_train, y_test, x_sensitive_train, x_sensitive_test =  train_test_split(X, y, x_sensitive, test_size=TEST_FOLD_SIZE, random_state=1234, shuffle=False)
    
    # show the data
    plot_data(x_test, y_test, x_sensitive_test, None, None, None, "img/data.png", None)





    """ Training the classifiers """
    # Classifier parameters 
    loss_function = "logreg" # perform the experiments with logistic regression
    EPS = 1e-4

    

    cons_params = {}
    cons_params["EPS"] = EPS
    cons_params["cons_type"] = -1 # no constraint



    print "\n\n== Unconstrained classifier =="
    # Train a classifier for each sensitive feature group separately optimizing accuracy for the respective group    
    clf_group = {} # will store the classifier for eah group here
    lam = {0:0.01, 1:0.01} # the regularization parameter -- we set small values here, in the paper, we cross validate all of regularization parameters
    for s_attr_val in set(x_sensitive_train):
        idx = x_sensitive_train==s_attr_val # the index for the current sensitive feature group
        clf = LinearClf(loss_function, lam=lam[s_attr_val], train_multiple=False) # for more info, see the clf documentation in the respective file
        clf.fit(x_train[idx], y_train[idx], x_sensitive_train[idx], cons_params)
        clf_group[s_attr_val] = clf

    # For simplicity of computing stats, we merge the two trained classifiers
    clf_merged = LinearClf(loss_function, lam=lam, train_multiple=True) 
    clf_merged.w = {0:None, 1:None}
    for s_attr_val in set(x_sensitive_train):
        clf_merged.w[s_attr_val] = clf_group[s_attr_val].w

    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf_merged, "img/unconstrained.png")



    
    print "\n\n== Parity classifier =="
    cons_params["cons_type"] = 0
    clf = LinearClf(loss_function, lam=0.01, train_multiple=False)
    clf.fit(x_train, y_train, x_sensitive_train, cons_params)
    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf, "img/parity.png")

    # compute the proxy value, will need this for the preferential classifiers
    dist_arr,dist_dict=clf.get_distance_boundary(x_train, x_sensitive_train)
    s_val_to_cons_sum_di = compute_stats.get_sensitive_attr_cov(dist_dict) # will need this for applying preferred fairness constraints



    
    print "\n\n\n\n== Preferred impact classifier =="


    # Not all values of the lambda satisfy the constraints empirically (in terms of acceptace rates)
    # This is because the scale (or norm) of the group-conditional classifiers can be very different from the baseline parity classifier, and from each other. This affects the distance from boundary (w.x) used in the constraints.
    # We use a hold out set with different regaularizer values to validate the norms that satisfy the constraints. Check the appendix of our NIPS paper for more details. 
    
    cons_params["cons_type"] = 1
    cons_params["tau"] = 1 # the tau value varies based on the dataset we look at. See the DCCP documentation for details
    cons_params["s_val_to_cons_sum"] = s_val_to_cons_sum_di
    lam = {0:2.5, 1:0.01} 
    clf = LinearClf(loss_function, lam=lam, train_multiple=True)
    clf.fit(x_train, y_train, x_sensitive_train, cons_params)
    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf, "img/preferred_impact.png")    


    
    print "\n\n\n\n== Preferred treatment AND preferred impact classifier =="
    cons_params["cons_type"] = 3
    cons_params["s_val_to_cons_sum"] = s_val_to_cons_sum_di
    lam = {0:2.5, 1:0.35}
    clf = LinearClf(loss_function, lam=lam, train_multiple=True)
    clf.fit(x_train, y_train, x_sensitive_train, cons_params)
    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf, "img/preferred_both.png")    
    


def main():
    test_synthetic_data()


if __name__ == '__main__':
    main()