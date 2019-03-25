import matplotlib.pyplot as plt
import numpy as np
from model import *
import json
from scipy.integrate import odeint

def plot_signals(t, signals, labels, starttime,stoptime, filename):

    xlim = [starttime,stoptime]
    num_signals = len(signals)

    motor_inds = []
    non_motor_inds = []
    for i in range(len(labels)):
        if "Motor" in labels[i]:
            motor_inds.append(i)
        elif "inh" in labels[i]:
            pass
        else:
            non_motor_inds.append(i)
    if len(motor_inds) == 0:
        if len(non_motor_inds) == 0:
            raise ValueError("Nothing to plot!")
        indices = [non_motor_inds]
    elif len(non_motor_inds) == 0:
        indices = [motor_inds]
    else:
        indices = [non_motor_inds, motor_inds]
    colors = ['k','r','g','b','y','m','xkcd:tomato','xkcd:lavender', 'xkcd:darkgreen', 'xkcd:plum', 'xkcd:salmon', 'xkcd:coral']

    figax = [plt.subplots(len(indices[i]), 1, figsize=(30, 2.5*len(indices[i])), facecolor='w', edgecolor='k') for i in range(len(indices))]
    figs = [figax[i][0] for i in range(len(figax))]
    axs = [figax[i][1].ravel() for i in range(len(figax))]
    for k in range(len(indices)):
        for i in range(len(indices[k])):
            j = indices[k][i]
            axs[k][i].plot(t, signals[j], colors[i],label=labels[j],linewidth = 3)
            axs[k][i].legend(loc = 1,fontsize = 25)
            axs[k][i].grid(True)
            ylim = 1 #(max(signals[i][xlim[0]:xlim[1]]) if max(signals[i])>0.1 else 1)
            axs[k][i].axis([xlim[0], xlim[1], 0, 1.1*ylim])
            if i != len(indices[k])-1:
                axs[k][i].set_xticklabels([])
            axs[k][i].tick_params(labelsize=25)

        figs[k].savefig("../img/" + filename + "_" + str(k+1))
#test
if __name__ == '__main__':

    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])
    t1 = params["t1"]
    t2 = params["t2"]
    amp = 0 #params["amp"]
    starttime = 5000
    stoptime = 60000
    res = model(b, c, vectorfield, t1, t2, amp, stoptime)
    t = res[0]
    signals = res[1:]
    labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF","Motor_HN", "Motor_PN", "Motor_VN","KF_inh", "NTS_inh"]
    filename = "test"
    plot_signals(t, signals, labels, starttime, stoptime, filename)