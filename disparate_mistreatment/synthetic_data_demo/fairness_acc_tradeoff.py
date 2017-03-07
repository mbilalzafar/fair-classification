import os,sys
import numpy as np
from generate_synthetic_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import funcs_disp_mist as fdm
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import plot_syn_boundaries as psb
from copy import deepcopy
import matplotlib.pyplot as plt # for plotting stuff

def test_synthetic_data():
	
	""" Generate the synthetic data """
	data_type = 1
	X, y, x_control = generate_synthetic_data(data_type=data_type, plot_data=False) # set plot_data to False to skip the data plot
	sensitive_attrs = x_control.keys()

	""" Split the data into train and test """
	train_fold_size = 0.5
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	cons_params = None # constraint parameters, will use them later
	loss_function = "logreg" # perform the experiments with logistic regression
	EPS = 1e-4

	def train_test_classifier():
		w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)

		train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)

		
		# accuracy and FPR are for the test because we need of for plotting
		# the covariance is for train, because we need it for setting the thresholds
		return w, test_score, s_attr_to_fp_fn_test, cov_all_train
		

	""" Classify the data while optimizing for accuracy """
	print
	print "== Unconstrained (original) classifier =="
	w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons, cov_all_train_uncons = train_test_classifier()
	print "\n-----------------------------------------------------------------------------------\n"

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	
	print
	print "== Classifier with fairness constraint =="

	it = 0.05
	mult_range = np.arange(1.0, 0.0-it, -it).tolist()


	acc_arr = []
	fpr_per_group = {0:[], 1:[]}
	fnr_per_group = {0:[], 1:[]}

	cons_type = 1 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2

	for m in mult_range:
		sensitive_attrs_to_cov_thresh = deepcopy(cov_all_train_uncons)
		for s_attr in sensitive_attrs_to_cov_thresh.keys():
			for cov_type in sensitive_attrs_to_cov_thresh[s_attr].keys():
				for s_val in sensitive_attrs_to_cov_thresh[s_attr][cov_type]:
					sensitive_attrs_to_cov_thresh[s_attr][cov_type][s_val] *= m


		cons_params = {"cons_type": cons_type, 
						"tau": tau, 
						"mu": mu, 
						"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

		w_cons, acc_cons, s_attr_to_fp_fn_test_cons, cov_all_train_cons  = train_test_classifier()
		
		fpr_per_group[0].append(s_attr_to_fp_fn_test_cons["s1"][0.0]["fpr"])
		fpr_per_group[1].append(s_attr_to_fp_fn_test_cons["s1"][1.0]["fpr"])
		fnr_per_group[0].append(s_attr_to_fp_fn_test_cons["s1"][0.0]["fnr"])
		fnr_per_group[1].append(s_attr_to_fp_fn_test_cons["s1"][1.0]["fnr"])


		acc_arr.append(acc_cons)


	fs = 15

	ax = plt.subplot(2,1,1)
	plt.plot(mult_range, fpr_per_group[0], "-o" , color="green", label = "Group-0")
	plt.plot(mult_range, fpr_per_group[1], "-o", color="blue", label = "Group-1")
	ax.set_xlim([max(mult_range), min(mult_range) ])
	plt.ylabel('False positive rate', fontsize=fs)
	ax.legend(fontsize=fs)
	

	ax = plt.subplot(2,1,2)
	plt.plot(mult_range, acc_arr, "-o" , color="green", label = "")
	ax.set_xlim([max(mult_range), min(mult_range) ])
	plt.xlabel('Covariance multiplicative factor (m)', fontsize=fs)
	plt.ylabel('Accuracy', fontsize=fs)

	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
	plt.savefig("img/fairness_acc_tradeoff_cons_type_%d.png" % cons_type)
	plt.show()


	return


def main():
	test_synthetic_data()


if __name__ == '__main__':
	main()