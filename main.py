from model import *
from plot_signals import plot_signals
import numpy as np
import json

# DEFINING PARAMETERS
num_nrns = 12
num_drives = 3

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - Relay
# 6 - NTS1  # 7 - NTS2    # 8 - KF
# 9 - M_HN  # 10- M_PN    # 11 - M_VN

file = open("rCPG_swCPG.json", "rb+")
params = json.load(file)
b = np.array(params["b"])
c = np.array(params["c"])

labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "KF","Motor_HN", "Motor_PN", "Motor_VN"]
stoptime = 60000
res = model(b, c, vectorfield, stoptime)
t = res[0]
signals = res[1:]
plot_signals(t,signals, labels, stoptime)