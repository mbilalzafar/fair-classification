import os,sys
import numpy as np
from generate_synthetic_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import funcs_disp_mist as fdm
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import plot_syn_boundaries as psb



def test_synthetic_data():
	
	""" Generate the synthetic data """
	data_type = 1
	X, y, x_control = generate_synthetic_data(data_type=data_type, plot_data=True) # set plot_data to False to skip the data plot
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
		return w, test_score, s_attr_to_fp_fn_test
		

	def plot_boundaries(w1, w2, p1, p2, acc1, acc2, fname):

		num_to_draw = 200 # we will only draw a small number of points to avoid clutter
		x_draw = X[:num_to_draw]
		y_draw = y[:num_to_draw]
		x_control_draw = x_control["s1"][:num_to_draw]

		X_s_0 = x_draw[x_control_draw == 0.0]
		X_s_1 = x_draw[x_control_draw == 1.0]
		y_s_0 = y_draw[x_control_draw == 0.0]
		y_s_1 = y_draw[x_control_draw == 1.0]
		plt.scatter(X_s_0[y_s_0==1.0][:, 1], X_s_0[y_s_0==1.0][:, 2], color='green', marker='x', s=30, linewidth=1.5)
		plt.scatter(X_s_0[y_s_0==-1.0][:, 1], X_s_0[y_s_0==-1.0][:, 2], color='red', marker='x', s=30, linewidth=1.5)
		plt.scatter(X_s_1[y_s_1==1.0][:, 1], X_s_1[y_s_1==1.0][:, 2], color='green', marker='o', facecolors='none', s=30)
		plt.scatter(X_s_1[y_s_1==-1.0][:, 1], X_s_1[y_s_1==-1.0][:, 2], color='red', marker='o', facecolors='none', s=30)


		x1,x2 = max(x_draw[:,1]), min(x_draw[:,1])
		y1,y2 = ut.get_line_coordinates(w1, x1, x2)
		plt.plot([x1,x2], [y1,y2], 'c-', linewidth=3, label = "Acc=%0.2f; p%% rule=%0.0f%% - Original"%(acc1, p1))
		y1,y2 = ut.get_line_coordinates(w2, x1, x2)
		plt.plot([x1,x2], [y1,y2], 'b--', linewidth=3, label = "Acc=%0.2f; p%% rule=%0.0f%% - Constrained"%(acc2, p2))



		plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
		plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
		plt.legend(loc=2, fontsize=15)
		plt.xlim((-15,10))
		plt.ylim((-10,15))
		plt.savefig(fname)
		plt.show()


	""" Classify the data while optimizing for accuracy """
	print
	print "== Unconstrained (original) classifier =="
	w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons = train_test_classifier()
	print "\n-----------------------------------------------------------------------------------\n"

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	
	print
	print "== Classifier with fairness constraint =="


	print "\n\n=== Constraints on FPR ==="	# setting parameter for constraints
	cons_type = 1 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {"s1": {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons  = train_test_classifier()
	psb.plot_boundaries(x_test, y_test, x_control_test, [w_uncons, w_cons], [acc_uncons, acc_cons], [s_attr_to_fp_fn_test_uncons["s1"], s_attr_to_fp_fn_test_cons["s1"]], "img/syn_cons_dtype_%d_cons_type_%d.png"%(data_type, cons_type) )
	print "\n-----------------------------------------------------------------------------------\n"

	print "\n\n=== Constraints on FNR ==="
	cons_type = 2
	cons_params["cons_type"] = cons_type # FNR constraint -- just change the cons_type, the rest of parameters should stay the same
	w_cons, acc_cons, s_attr_to_fp_fn_test_cons  = train_test_classifier()
	psb.plot_boundaries(x_test, y_test, x_control_test, [w_uncons, w_cons], [acc_uncons, acc_cons], [s_attr_to_fp_fn_test_uncons["s1"], s_attr_to_fp_fn_test_cons["s1"]], "img/syn_cons_dtype_%d_cons_type_%d.png"%(data_type, cons_type) )
	print "\n-----------------------------------------------------------------------------------\n"



	print "\n\n=== Constraints on both FPR and FNR ==="
	cons_type = 4
	cons_params["cons_type"] = cons_type # both FPR and FNR constraints
	w_cons, acc_cons, s_attr_to_fp_fn_test_cons  = train_test_classifier()
	psb.plot_boundaries(x_test, y_test, x_control_test, [w_uncons, w_cons], [acc_uncons, acc_cons], [s_attr_to_fp_fn_test_uncons["s1"], s_attr_to_fp_fn_test_cons["s1"]], "img/syn_cons_dtype_%d_cons_type_%d.png"%(data_type, cons_type) )
	print "\n-----------------------------------------------------------------------------------\n"


	return


def main():
	test_synthetic_data()


if __name__ == '__main__':
	main()