from model import *
from plot_signals import plot_signals
import numpy as np
import json

# DEFINING PARAMETERS
num_nrns = 15
num_drives = 3

file = open("rCPG_swCPG.json", "rb+")
params = json.load(file)
b = np.array(params["b"])
c = np.array(params["c"])


labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF","Motor_HN", "Motor_PN", "Motor_VN","KF_inh", "NTS_inh"]
stoptime = 60000

t_start = [25000 + 250*i for i in range(5)]
t_stop = [35000 + 250*i for i in range(5)]
for i in range(len(t_start)):
    t1 = t_start[i]
    t2 = t_stop[i]
    file_name = "Normal_swallowing_" +str(t1) +"_"+ str(t2)
    res = model(b, c, vectorfield, t1, t2, stoptime)
    t = res[0]
    signals = res[1:]
    plot_signals(t,signals[:-2], labels, stoptime, file_name)