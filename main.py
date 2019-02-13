from model import *
from plot_signals import plot_signals
import numpy as np
import json

# DEFINING PARAMETERS
num_nrns = 13
num_drives = 3

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - EarlyI2
# 6 - Relay # 7 - NTS1    # 8 - NTS2
# 9 - KF #  # 10 - M_HN   # 11- M_PN
# 12 - M_VN

file = open("rCPG_swCPG.json", "rb+")
params = json.load(file)
b = np.array(params["b"])
c = np.array(params["c"])

labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "EarlyI2", "Relay", "NTS1", "NTS2", "KF","Motor_HN", "Motor_PN", "Motor_VN"]
stoptime = 60000
res = model(b, c, vectorfield, stoptime)
t = res[0]
signals = res[1:]
plot_signals(t,signals, labels, stoptime)