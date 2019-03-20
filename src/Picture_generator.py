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

# Different Aplitudes
amps = [110 + 10*i for i in range(30) ]
for i in range(len(amps)):
    t1 = 25000
    t2 = 35000
    amp = amps[i]
    file_name = "Normal_swallowing_" + str(amp)
    res = model(b, c, vectorfield, t1, t2, amp, stoptime)
    t = res[0]
    signals = res[1:]


    # plot_signals(t,signals[:-2], labels, stoptime, file_name)

# t_start = [25000 + 250*i for i in range(7)]
# t_stop = [25100 + 250*i for i in range(7)]
