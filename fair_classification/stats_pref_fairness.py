from __future__ import division
import numpy as np
from sklearn.preprocessing import MaxAbsScaler # normalize data with 0 and 1 as min/max absolute vals
import scipy
from multiprocessing import Pool, Process, Queue
from sklearn.metrics import roc_auc_score
import traceback


def get_acc_all(dist_arr, y):
    """
    Get accuracy for all data points
    Each group gets the prediction based on their own boundary
    """
    return np.sum(sign_bin_clf(dist_arr) == y) / y.shape[0]

def get_clf_stats(dist_arr, dist_dict, y, x_sensitive, print_stats=False):


    # compute the class labels
    all_class_labels_assigned = sign_bin_clf(dist_arr)
    
    
    s_val_to_cons_sum = {}
        

    acc = get_acc_all(dist_arr,y)

    if print_stats:
        print "\n\n\nAccuracy: %0.3f\n" % acc

        
    acc_stats = get_acc_stats(dist_dict, y, x_sensitive, print_stats)
    s_val_to_cons_sum = get_sensitive_attr_cov(dist_dict) 

    return acc, s_val_to_cons_sum, acc_stats


def get_fp_fn_tp_tn(y_true, y_pred):
    

    def check_labels_bin(arr):

        """ Can only have -1 and 1"""
        try:
            if len(set(arr)) == 1:
                elem = list(set(arr))[0]
                assert(elem==1 or elem==-1)
            else:
                assert(len(set(arr)) == 2)
                assert( sorted(list(set(arr)))[0] == -1 and sorted(list(set(arr)))[1] == 1 )
        except:
            traceback.print_exc()
            raise Exception("Class labels (both true and predicted) can only take values -1 and 1... Exiting...")
            
    
    check_labels_bin(y_true)
    check_labels_bin(y_pred)


    
    acc = float(sum(y_true==y_pred)) / len(y_true)

    fp = sum(np.logical_and(y_true == -1.0, y_pred == +1.0)) # something which is -ve but is misclassified as +ve
    fn = sum(np.logical_and(y_true == +1.0, y_pred == -1.0)) # something which is +ve but is misclassified as -ve
    tp = sum(np.logical_and(y_true == +1.0, y_pred == +1.0)) # something which is +ve AND is correctly classified as +ve
    tn = sum(np.logical_and(y_true == -1.0, y_pred == -1.0)) # something which is -ve AND is correctly classified as -ve

    fpr = float(fp) / float(fp + tn)
    fnr = float(fn) / float(fn + tp)
    tpr = float(tp) / float(tp + fn)
    tnr = float(tn) / float(tn + fp)
    frac_pos = (tp + fp) / (tp + tn + fp + fn) # fraction classified as positive

    out_dict = {"fpr": fpr, "fnr": fnr, "acc": acc, "frac_pos": frac_pos}

    return out_dict



def get_acc_stats(dist_dict, y, x_sensitive, verbose = False):


    """
    output dict form: s_attr_group (0/1) -> w_group (0/1) -> fpr/fnr/acc/frac_pos
    """

    acc_stats = {}

    try:            
        assert(len(set(x_sensitive)) == 2)        
    except:
        raise Exception("Fill the constraint code for categorical sensitive features... Exiting...")

    try:
        assert( sorted(list(set(x_sensitive)))[0] == 0 and sorted(list(set(x_sensitive)))[1] == 1 )
    except:
        raise Exception("Sensitive feature can only take values 0 and 1... Exiting...")
        



    if verbose == True:
        print "||  s  ||   frac_pos  ||"


    for s_val in set(x_sensitive):
        idx = x_sensitive == s_val
        other_val = np.abs(1-s_val) 
        acc_stats[s_val] = {}


        y_true_local = y[idx]
        y_pred_local = sign_bin_clf(dist_dict[s_val][s_val]) # predictions with this classifier
        y_pred_local_other = sign_bin_clf(dist_dict[s_val][other_val])  # predictions with other group's classifier

        
        assert(y_true_local.shape[0] == y_pred_local.shape[0] and y_true_local.shape[0] == y_pred_local_other.shape[0])


        acc_stats[s_val][s_val] = get_fp_fn_tp_tn(y_true_local, y_pred_local)
        acc_stats[s_val][other_val] = get_fp_fn_tp_tn(y_true_local, y_pred_local_other)


        if verbose == True:
            if isinstance(s_val, float): # print the int value of the sensitive attr val
                s_val = int(s_val)


            print "||  %s  || %0.2f (%0.2f) ||" % (s_val, acc_stats[s_val][s_val]["frac_pos"], acc_stats[s_val][other_val]["frac_pos"])
                


    return acc_stats            



def sign_bin_clf(arr):
    """
        prediction for a linear classifier. np.sign gives 0 for sing(0), we want 1

        if arr[i] >= 0, arr[i] = +1
        else arr[i] = -1
        
    """
    arr = np.sign(arr)
    arr[arr==0] = 1
    return arr


def get_sensitive_attr_cov(dist_dict):

    """
    computes the ramp function for each group to estimate the acceptance rate
    """    

    s_val_to_cons_sum = {0:{}, 1:{}} # s_attr_group (0/1) -> w_group (0/1) -> ramp approx
    
    for s_val in dist_dict.keys():
        for w_group in dist_dict[s_val].keys():
            fx = dist_dict[s_val][w_group]            
            s_val_to_cons_sum[s_val][w_group] = np.sum( np.maximum(0, fx) ) / fx.shape[0]
            

    return s_val_to_cons_sum







def add_intercept(x):

    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)



def scale_data(x_train, x_test):

    """
        We only scale the continuous features. No need to scale binary features
    """

    
    idx_binary = [] # columns with boolean values
    for k in range(x_train.shape[1]):
        idx_binary.append( np.array_equal(x_train[:,k], x_train[:,k].astype(bool)) ) # checking if a column is binary
    idx_cont = np.logical_not(idx_binary)


    sc = MaxAbsScaler()
    sc.fit(x_train[:, idx_cont])
    
    x_train[:, idx_cont] = sc.transform(x_train[:, idx_cont])
    x_test[:, idx_cont] = sc.transform(x_test[:, idx_cont])

    return

