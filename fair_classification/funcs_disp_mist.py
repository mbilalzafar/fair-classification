from __future__ import division
import os,sys
import traceback
import numpy as np
from random import seed, shuffle
from collections import defaultdict
from copy import deepcopy
from cvxpy import *
import dccp
from dccp.problem import is_dccp
import utils as ut

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def train_model_disp_mist(x, y, x_control, loss_function, EPS, cons_params=None):

    # cons_type, sensitive_attrs_to_cov_thresh, take_initial_sol, gamma, tau, mu, EPS, cons_type
    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "disparate_mistreatment/synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    EPS: stopping criteria for the convex solver. check the CVXPY documentation for details. default for CVXPY is 1e-6

    cons_params: is None when we do not want to apply any constraints
    otherwise: cons_params is a dict with keys as follows:
        - cons_type: 
            - 0 for all misclassifications 
            - 1 for FPR
            - 2 for FNR
            - 4 for both FPR and FNR
        - tau: DCCP parameter, controls how much weight to put on the constraints, if the constraints are not satisfied, then increase tau -- default is DCCP val 0.005
        - mu: DCCP parameter, controls the multiplicative factor by which the tau increases in each DCCP iteration -- default is the DCCP val 1.2
        - take_initial_sol: whether the starting point for DCCP should be the solution for the original (unconstrained) classifier -- default value is True
        - sensitive_attrs_to_cov_thresh: covariance threshold for each cons_type, eg, key 1 contains the FPR covariance
    ----

    Outputs:

    w: the learned weight vector for the classifier

    """

    max_iters = 100 # for the convex program
    max_iter_dccp = 50  # for the dccp algo

    
    num_points, num_features = x.shape
    w = Variable(num_features) # this is the weight vector

    # initialize a random value of w
    np.random.seed(112233)
    w.value = np.random.rand(x.shape[1])

    if cons_params is None: # just train a simple classifier, no fairness constraints
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, cons_params["sensitive_attrs_to_cov_thresh"], cons_params["cons_type"], w)


    if loss_function == "logreg":
        # constructing the logistic loss problem
        loss = sum_entries(  logistic( mul_elemwise(-y, x*w) )  ) / num_points # we are converting y to a diagonal matrix for consistent


    # sometimes, its a good idea to give a starting point to the constrained solver
    # this starting point for us is the solution to the unconstrained optimization problem
    # another option of starting point could be any feasible solution
    if cons_params is not None:
        if cons_params.get("take_initial_sol") is None: # true by default
            take_initial_sol = True
        elif cons_params["take_initial_sol"] == False:
            take_initial_sol = False

        if take_initial_sol == True: # get the initial solution
            p = Problem(Minimize(loss), [])
            p.solve()


    # construct the cvxpy problem
    prob = Problem(Minimize(loss), constraints)

    # print "\n\n"
    # print "Problem is DCP (disciplined convex program):", prob.is_dcp()
    # print "Problem is DCCP (disciplined convex-concave program):", is_dccp(prob)

    try:

        tau, mu = 0.005, 1.2 # default dccp parameters, need to be varied per dataset
        if cons_params is not None: # in case we passed these parameters as a part of dccp constraints
            if cons_params.get("tau") is not None: tau = cons_params["tau"]
            if cons_params.get("mu") is not None: mu = cons_params["mu"]

        prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
            solver=ECOS, verbose=False, 
            feastol=EPS, abstol=EPS, reltol=EPS,feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
            max_iters=max_iters, max_iter=max_iter_dccp)

        
        assert(prob.status == "Converged" or prob.status == "optimal")
        # print "Optimization done, problem status:", prob.status

    except:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


    # check that the fairness constraint is satisfied
    for f_c in constraints:
        assert(f_c.value == True) # can comment this out if the solver fails too often, but make sure that the constraints are satisfied empirically. alternatively, consider increasing tau parameter
        pass
        

    w = np.array(w.value).flatten() # flatten converts it to a 1d array


    return w


def get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs):



    
    assert(len(sensitive_attrs) == 1) # ensure that we have just one sensitive attribute
    s_attr = sensitive_attrs[0] # for now, lets compute the accuracy for just one sensitive attr


    # compute distance from boundary
    distances_boundary_train = get_distance_boundary(w, x_train, x_control_train[s_attr])
    distances_boundary_test = get_distance_boundary(w, x_test, x_control_test[s_attr])

    # compute the class labels
    all_class_labels_assigned_train = np.sign(distances_boundary_train)
    all_class_labels_assigned_test = np.sign(distances_boundary_test)


    train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(None, x_train, y_train, x_test, y_test, all_class_labels_assigned_train, all_class_labels_assigned_test)

    
    cov_all_train = {}
    cov_all_test = {}
    for s_attr in sensitive_attrs:
        
        
        print_stats = False # we arent printing the stats for the train set to avoid clutter

        # uncomment these lines to print stats for the train fold
        # print "*** Train ***"
        # print "Accuracy: %0.3f" % (train_score)
        # print_stats = True
        s_attr_to_fp_fn_train = get_fpr_fnr_sensitive_features(y_train, all_class_labels_assigned_train, x_control_train, sensitive_attrs, print_stats)
        cov_all_train[s_attr] = get_sensitive_attr_constraint_fpr_fnr_cov(None, x_train, y_train, distances_boundary_train, x_control_train[s_attr]) 
        

        print "\n"
        print "Accuracy: %0.3f" % (test_score)
        print_stats = True # only print stats for the test fold
        s_attr_to_fp_fn_test = get_fpr_fnr_sensitive_features(y_test, all_class_labels_assigned_test, x_control_test, sensitive_attrs, print_stats)
        cov_all_test[s_attr] = get_sensitive_attr_constraint_fpr_fnr_cov(None, x_test, y_test, distances_boundary_test, x_control_test[s_attr]) 
        print "\n"

    return train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test

def get_distance_boundary(w, x, s_attr_arr):

    """
        if we have boundaries per group, then use those separate boundaries for each sensitive group
        else, use the same weight vector for everything
    """

    distances_boundary = np.zeros(x.shape[0])
    if isinstance(w, dict): # if we have separate weight vectors per group
        for k in w.keys(): # for each w corresponding to each sensitive group
            d = np.dot(x, w[k])
            distances_boundary[s_attr_arr == k] = d[s_attr_arr == k] # set this distance only for people with this sensitive attr val
    else: # we just learn one w for everyone else
        distances_boundary = np.dot(x, w)
    return distances_boundary


def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs_to_cov_thresh, cons_type, w):

    """
    get the list of constraints to be fed to the minimizer

    cons_type == 0: means the whole combined misclassification constraint (without FNR or FPR)
    cons_type == 1: FPR constraint
    cons_type == 2: FNR constraint
    cons_type == 4: both FPR as well as FNR constraints

    sensitive_attrs_to_cov_thresh: is a dict like {s: {cov_type: val}}
    s is the sensitive attr
    cov_type is the covariance type. contains the covariance for all misclassifications, FPR and for FNR etc
    """

    constraints = []
    for attr in sensitive_attrs_to_cov_thresh.keys():

        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = ut.get_one_hot_encoding(attr_arr)
                
        if index_dict is None: # binary attribute, in this case, the attr_arr_transformed is the same as the attr_arr

            s_val_to_total = {ct:{} for ct in [0,1,2]} # constrain type -> sens_attr_val -> total number
            s_val_to_avg = {ct:{} for ct in [0,1,2]}
            cons_sum_dict = {ct:{} for ct in [0,1,2]} # sum of entities (females and males) in constraints are stored here

            for v in set(attr_arr):
                s_val_to_total[0][v] = sum(x_control_train[attr] == v)
                s_val_to_total[1][v] = sum(np.logical_and(x_control_train[attr] == v, y_train == -1)) # FPR constraint so we only consider the ground truth negative dataset for computing the covariance
                s_val_to_total[2][v] = sum(np.logical_and(x_control_train[attr] == v, y_train == +1))


            for ct in [0,1,2]:
                s_val_to_avg[ct][0] = s_val_to_total[ct][1] / float(s_val_to_total[ct][0] + s_val_to_total[ct][1]) # N1/N in our formulation, differs from one constraint type to another
                s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0] # N0/N

            
            for v in set(attr_arr):

                idx = x_control_train[attr] == v                


                #################################################################
                # #DCCP constraints
                dist_bound_prod = mul_elemwise(y_train[idx], x_train[idx] * w) # y.f(x)
                
                cons_sum_dict[0][v] = sum_entries( min_elemwise(0, dist_bound_prod) ) * (s_val_to_avg[0][v] / len(x_train)) # avg misclassification distance from boundary
                cons_sum_dict[1][v] = sum_entries( min_elemwise(0, mul_elemwise( (1 - y_train[idx])/2.0, dist_bound_prod) ) ) * (s_val_to_avg[1][v] / sum(y_train == -1)) # avg false positive distance from boundary (only operates on the ground truth neg dataset)
                cons_sum_dict[2][v] = sum_entries( min_elemwise(0, mul_elemwise( (1 + y_train[idx])/2.0, dist_bound_prod) ) ) * (s_val_to_avg[2][v] / sum(y_train == +1)) # avg false negative distance from boundary
                #################################################################

                
            if cons_type == 4:
                cts = [1,2]
            elif cons_type in [0,1,2]:
                cts = [cons_type]
            
            else:
                raise Exception("Invalid constraint type")


            #################################################################
            #DCCP constraints
            for ct in cts:
                thresh = abs(sensitive_attrs_to_cov_thresh[attr][ct][1] - sensitive_attrs_to_cov_thresh[attr][ct][0])
                constraints.append( cons_sum_dict[ct][1] <= cons_sum_dict[ct][0]  + thresh )
                constraints.append( cons_sum_dict[ct][1] >= cons_sum_dict[ct][0]  - thresh )

            #################################################################


            
        else: # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately
            # need to fill up this part
            raise Exception("Fill the constraint code for categorical sensitive features... Exiting...")
            sys.exit(1)
            

    return constraints


def get_fpr_fnr_sensitive_features(y_true, y_pred, x_control, sensitive_attrs, verbose = False):



    # we will make some changes to x_control in this function, so make a copy in order to preserve the origianl referenced object
    x_control_internal = deepcopy(x_control)

    s_attr_to_fp_fn = {}
    
    for s in sensitive_attrs:
        s_attr_to_fp_fn[s] = {}
        s_attr_vals = x_control_internal[s]
        if verbose == True:
            print "||  s  || FPR. || FNR. ||"
        for s_val in sorted(list(set(s_attr_vals))):
            s_attr_to_fp_fn[s][s_val] = {}
            y_true_local = y_true[s_attr_vals==s_val]
            y_pred_local = y_pred[s_attr_vals==s_val]

            

            acc = float(sum(y_true_local==y_pred_local)) / len(y_true_local)

            fp = sum(np.logical_and(y_true_local == -1.0, y_pred_local == +1.0)) # something which is -ve but is misclassified as +ve
            fn = sum(np.logical_and(y_true_local == +1.0, y_pred_local == -1.0)) # something which is +ve but is misclassified as -ve
            tp = sum(np.logical_and(y_true_local == +1.0, y_pred_local == +1.0)) # something which is +ve AND is correctly classified as +ve
            tn = sum(np.logical_and(y_true_local == -1.0, y_pred_local == -1.0)) # something which is -ve AND is correctly classified as -ve

            all_neg = sum(y_true_local == -1.0)
            all_pos = sum(y_true_local == +1.0)

            fpr = float(fp) / float(fp + tn)
            fnr = float(fn) / float(fn + tp)
            tpr = float(tp) / float(tp + fn)
            tnr = float(tn) / float(tn + fp)


            s_attr_to_fp_fn[s][s_val]["fp"] = fp
            s_attr_to_fp_fn[s][s_val]["fn"] = fn
            s_attr_to_fp_fn[s][s_val]["fpr"] = fpr
            s_attr_to_fp_fn[s][s_val]["fnr"] = fnr

            s_attr_to_fp_fn[s][s_val]["acc"] = (tp + tn) / (tp + tn + fp + fn)
            if verbose == True:
                if isinstance(s_val, float): # print the int value of the sensitive attr val
                    s_val = int(s_val)
                print "||  %s  || %0.2f || %0.2f ||" % (s_val, fpr, fnr)

        
        return s_attr_to_fp_fn


def get_sensitive_attr_constraint_fpr_fnr_cov(model, x_arr, y_arr_true, y_arr_dist_boundary, x_control_arr, verbose=False):

    
    """
    Here we compute the covariance between sensitive attr val and ONLY misclassification distances from boundary for False-positives
    (-N_1 / N) sum_0(min(0, y.f(x))) + (N_0 / N) sum_1(min(0, y.f(x))) for all misclassifications
    (-N_1 / N) sum_0(min(0, (1-y)/2 . y.f(x))) + (N_0 / N) sum_1(min(0,  (1-y)/2. y.f(x))) for FPR

    y_arr_true are the true class labels
    y_arr_dist_boundary are the predicted distances from the decision boundary

    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function


    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

        
    assert(x_arr.shape[0] == x_control_arr.shape[0])
    if len(x_control_arr.shape) > 1: # make sure we just have one column in the array
        assert(x_control_arr.shape[1] == 1)
    if len(set(x_control_arr)) != 2: # non binary attr
        raise Exception("Non binary attr, fix to handle non bin attrs")

    
    arr = []
    if model is None:
        arr = y_arr_dist_boundary * y_arr_true # simply the output labels
    else:
        arr = np.dot(model, x_arr.T) * y_arr_true # the product with the weight vector -- the sign of this is the output label
    arr = np.array(arr)

    s_val_to_total = {ct:{} for ct in [0,1,2]}
    s_val_to_avg = {ct:{} for ct in [0,1,2]}
    cons_sum_dict = {ct:{} for ct in [0,1,2]} # sum of entities (females and males) in constraints are stored here

    for v in set(x_control_arr):
        s_val_to_total[0][v] = sum(x_control_arr == v)
        s_val_to_total[1][v] = sum(np.logical_and(x_control_arr == v, y_arr_true == -1))
        s_val_to_total[2][v] = sum(np.logical_and(x_control_arr == v, y_arr_true == +1))


    for ct in [0,1,2]:
        s_val_to_avg[ct][0] = s_val_to_total[ct][1] / float(s_val_to_total[ct][0] + s_val_to_total[ct][1]) # N1 / N
        s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0] # N0 / N

    
    for v in set(x_control_arr):
        idx = x_control_arr == v
        dist_bound_prod = arr[idx]

        cons_sum_dict[0][v] = sum( np.minimum(0, dist_bound_prod) ) * (s_val_to_avg[0][v] / len(x_arr))
        cons_sum_dict[1][v] = sum( np.minimum(0, ( (1 - y_arr_true[idx]) / 2 ) * dist_bound_prod) ) * (s_val_to_avg[1][v] / sum(y_arr_true == -1))
        cons_sum_dict[2][v] = sum( np.minimum(0, ( (1 + y_arr_true[idx]) / 2 ) * dist_bound_prod) ) * (s_val_to_avg[2][v] / sum(y_arr_true == +1))
        

    cons_type_to_name = {0:"ALL", 1:"FPR", 2:"FNR"}
    for cons_type in [0,1,2]:
        cov_type_name = cons_type_to_name[cons_type]    
        cov = cons_sum_dict[cons_type][1] - cons_sum_dict[cons_type][0]
        if verbose == True:
            print "Covariance for type '%s' is: %0.7f" %(cov_type_name, cov)
        
    return cons_sum_dict
    

def plot_fairness_acc_tradeoff(x_all, y_all, x_control_all, loss_function, cons_type):


    # very the covariance threshold using a range of decreasing multiplicative factors and see the tradeoffs between accuracy and fairness
    it = 0.2
    mult_range = np.arange(1.0, 0.0-it, -it).tolist()

    


    positive_class_label = 1 # positive class is +1
    test_acc = []
    

    # first get the original values of covariance in the unconstrained classifier -- these original values are not needed for reverse constraint    
    test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, 0, apply_accuracy_constraint, sep_constraint, sensitive_attrs, [{} for i in range(0,num_folds)], 0)

    for c in cov_range:
        print "LOG: testing for multiplicative factor: %0.2f" % c
        sensitive_attrs_to_cov_original_arr_multiplied = []
        for sensitive_attrs_to_cov_original in cov_dict_train_arr:
            sensitive_attrs_to_cov_thresh = deepcopy(sensitive_attrs_to_cov_original)
            for k in sensitive_attrs_to_cov_thresh.keys():
                v = sensitive_attrs_to_cov_thresh[k]
                if type(v) == type({}):
                    for k1 in v.keys():
                        v[k1] = v[k1] * c
                else:
                    sensitive_attrs_to_cov_thresh[k] = v * c
            sensitive_attrs_to_cov_original_arr_multiplied.append(sensitive_attrs_to_cov_thresh)


        test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr  = compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_original_arr_multiplied, c)
        test_acc.append(np.mean(test_acc_arr))


        correlation_dict_train = get_avg_correlation_dict(correlation_dict_train_arr)
        correlation_dict_test = get_avg_correlation_dict(correlation_dict_test_arr)
        
        # just plot the correlations for the first sensitive attr, the plotting can be extended for the other values, but as a proof of concept, we will jsut show for one
        s = sensitive_attrs[0]    
        
        for k,v in correlation_dict_test[s].items():
            if v.get(positive_class_label) is None:
                positive_per_category[k].append(0.0)
            else:
                positive_per_category[k].append(v[positive_class_label])
    
    positive_per_category = dict(positive_per_category)
    
    p_rule_arr = (np.array(positive_per_category[0]) / np.array(positive_per_category[1])) * 100.0
    

    ax = plt.subplot(2,1,1)
    plt.plot(cov_range, positive_per_category[0], "-o" , color="green", label = "Protected")
    plt.plot(cov_range, positive_per_category[1], "-o", color="blue", label = "Non-protected")
    ax.set_xlim([min(cov_range), max(cov_range)])
    plt.xlabel('Multiplicative loss factor')
    plt.ylabel('Perc. in positive class')
    if apply_accuracy_constraint == False:
        plt.gca().invert_xaxis()
        plt.xlabel('Multiplicative covariance factor (c)')
    ax.legend()

    ax = plt.subplot(2,1,2)
    plt.scatter(p_rule_arr, test_acc, color="red")
    ax.set_xlim([min(p_rule_arr), max(max(p_rule_arr), 100)])
    plt.xlabel('P% rule')
    plt.ylabel('Accuracy')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()


