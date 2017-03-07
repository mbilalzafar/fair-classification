import matplotlib
import matplotlib.pyplot as plt # for plotting stuff
import os

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'figure.autolayout': True})

def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]    
    return y1,y2
def plot_boundaries(X, y, x_control, w_arr, acc_arr, fp_fn_arr, fname):




	# print fp_fn_arr
	plt.figure()
	num_to_draw = 200 # we will only draw a small number of points to avoid clutter

	x_draw = X[:num_to_draw]
	y_draw = y[:num_to_draw]
	x_control_draw = x_control["s1"][:num_to_draw]

	X_s_0 = x_draw[x_control_draw == 0.0]
	X_s_1 = x_draw[x_control_draw == 1.0]
	y_s_0 = y_draw[x_control_draw == 0.0]
	y_s_1 = y_draw[x_control_draw == 1.0]
	plt.scatter(X_s_0[y_s_0==1.0][:, 1], X_s_0[y_s_0==1.0][:, 2], color='green', marker='x', s=60, linewidth=2)
	plt.scatter(X_s_0[y_s_0==-1.0][:, 1], X_s_0[y_s_0==-1.0][:, 2], color='red', marker='x', s=60, linewidth=2)
	plt.scatter(X_s_1[y_s_1==1.0][:, 1], X_s_1[y_s_1==1.0][:, 2], color='green', marker='o', facecolors='none', s=60, linewidth=2)
	plt.scatter(X_s_1[y_s_1==-1.0][:, 1], X_s_1[y_s_1==-1.0][:, 2], color='red', marker='o', facecolors='none', s=60, linewidth=2)


	assert(len(w_arr) == len(acc_arr))
	assert(len(w_arr) == len(fp_fn_arr))

	line_styles = ['c-', 'b--', 'k--', 'c--', 'b--']


	for i in range(0,len(w_arr)):
		x1,x2 = max(x_draw[:,1])-2, min(x_draw[:,1])
		y1,y2 = get_line_coordinates(w_arr[i], x1, x2)

		

		l = "Acc=%0.2f; FPR=%0.2f:%0.2f; FNR=%0.2f:%0.2f"%(acc_arr[i], fp_fn_arr[i][0.0]["fpr"], fp_fn_arr[i][1.0]["fpr"], fp_fn_arr[i][0.0]["fnr"], fp_fn_arr[i][1.0]["fnr"])
		plt.plot([x1,x2], [y1,y2], line_styles[i], linewidth=5, label = l)




	plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
	plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
	plt.legend(loc=2, fontsize=21)
	
	plt.ylim((-8,12))
	
	plt.savefig(fname)


	plt.show()