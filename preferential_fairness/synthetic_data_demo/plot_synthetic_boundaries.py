import matplotlib
import matplotlib.pyplot as plt # for plotting stuff
import os
import numpy as np

matplotlib.rcParams['text.usetex'] = True # for type-1 fonts

def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]    
    return y1,y2

def plot_data(X, y, x_sensitive, w_arr, label_arr, lt_arr, fname, title, group=None):


    # print fp_fn_arr
    plt.figure()
    num_to_draw = 200 # we will only draw a small number of points to avoid clutter
    fs = 20 # font size for labels and legends

    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    x_sensitive_draw = x_sensitive[:num_to_draw]


    x_lim = [min(x_draw[:,-2]) - np.absolute(0.3*min(x_draw[:,-2])), max(x_draw[:,-2]) + np.absolute(0.5 * max(x_draw[:,-2]))]
    y_lim = [min(x_draw[:,-1]) - np.absolute(0.3*min(x_draw[:,-1])), max(x_draw[:,-1]) + np.absolute(0.7 * max(x_draw[:,-1]))]

    X_s_0 = x_draw[x_sensitive_draw == 0.0]
    X_s_1 = x_draw[x_sensitive_draw == 1.0]
    y_s_0 = y_draw[x_sensitive_draw == 0.0]
    y_s_1 = y_draw[x_sensitive_draw == 1.0]

    if w_arr is not None: # we are plotting the boundaries of a trained classifier
        plt.scatter(X_s_0[y_s_0==1.0][:, -2], X_s_0[y_s_0==1.0][:, -1], color='green', marker='x', s=70, linewidth=2)
        plt.scatter(X_s_0[y_s_0==-1.0][:, -2], X_s_0[y_s_0==-1.0][:, -1], color='red', marker='x', s=70, linewidth=2)
        plt.scatter(X_s_1[y_s_1==1.0][:, -2], X_s_1[y_s_1==1.0][:, -1], color='green', marker='o', facecolors='none', s=70, linewidth=2)
        plt.scatter(X_s_1[y_s_1==-1.0][:, -2], X_s_1[y_s_1==-1.0][:, -1], color='red', marker='o', facecolors='none', s=70, linewidth=2)


        for i in range(0, len(w_arr)):
            w = w_arr[i]
            l = label_arr[i]
            lt = lt_arr[i]

            x1,x2 = min(x_draw[:,1]), max(x_draw[:,1])
            y1,y2 = get_line_coordinates(w, x1, x2)

            plt.plot([x1,x2], [y1,y2], lt, linewidth=3, label = l)


        plt.title(title, fontsize=fs)

    else: # just plotting the data
        plt.scatter(X_s_0[y_s_0==1.0][:, -2], X_s_0[y_s_0==1.0][:, -1], color='green', marker='x', s=70, linewidth=2, label= "group-0 +ve")
        plt.scatter(X_s_0[y_s_0==-1.0][:, -2], X_s_0[y_s_0==-1.0][:, -1], color='red', marker='x', s=70, linewidth=2, label= "group-0 -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, -2], X_s_1[y_s_1==1.0][:, -1], color='green', marker='o', facecolors='none', s=70, linewidth=2, label= "group-1 +ve")
        plt.scatter(X_s_1[y_s_1==-1.0][:, -2], X_s_1[y_s_1==-1.0][:, -1], color='red', marker='o', facecolors='none', s=70, linewidth=2, label= "group-1 -ve")


    if True: # turn the ticks on or off
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(loc=2, fontsize=fs)
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    

    plt.savefig(fname)
    

    plt.show()



