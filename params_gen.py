import json
import numpy as np

params = dict()

num_nrns = 13
num_drives = 3
x = 1.0
y = 1.0 #1.1 0.7 - partial recovery

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - EarlyI2
# 6 - Relay # 7 - NTS1    # 8 - NTS2
# 9 - KF #  # 10 - M_HN   # 11- M_PN
# 12 - M_VN

b = np.zeros((num_nrns, num_nrns))
# positive weights
b[0,0] = 0.3  #PreI -> PreI
b[0,1] = 0.6  #PreI -> EarlyI
b[0,4] = 0.9  #PreI -> RampI
b[7,0] = 0.3 # NTS1 -> PreI
b[7,1] = 0.4 # NTS1 -> EarlyI1
# b[7,2] = 0.0 # NTS1 -> PostI
# b[8,2] = 0.0 # NTS2 -> PostI
b[7,3] = 0.1 # NTS1 -> AugE
b[8,3] = 0.1 # NTS2 -> AugE
b[9,2] = 1.7 # KF -> PostI
b[6,7] = 0.2 # Relay -> NTS1
b[6,8] = 1.2 # Relay -> NTS2
b[6,9] = 2.0 # Relay -> KF
b[0,10] = 1.5 # PreI -> M_HN
b[8,10] = 0.9 # NTS2 -> M_HN
b[4,11] = 2.8 # RampI -> M_PN
b[2,12] = 1.7 # PostI -> M_VN
b[4,12] = 0.9 # RampI -> M_VN
b[8,12] = 1.8 # NTS2 - M_VN
b[9,8]  =  0.22 #KF -> NTS2
b[9,7]  =  0.22 #KF -> NTS1
# negative weights
b[1,2] = -0.15 #EarlyI -> PostI
b[1,3] = -0.39 #EarlyI -> AugE

b[1,9] = -0.02 #EarlyI -> KF #!!!

b[2,0] = -0.2 #PostI -> PreI
b[2,1] = -0.1 #PostI -> EarlyI
b[2,3] = -0.2 #PostI -> AugE
b[2,4] = -0.4  #PostI -> RampI
b[2,5] = -0.1  #PostI -> EarlyI2
b[3,4] = -0.2  #AugE -> RampI
b[3,0] = -0.32  #AugE -> PreI
b[3,1] = -0.35  #AugE -> EarlyI
b[3,2] = -0.1  #AugE -> PostI
# b[5,4] = -0.4  #EarlyI2 -> RampI
b[1,4] = -0.15  #EarlyI1 -> RampI
# b[7,0] = -0.1 #NTS1 -> PreI
# b[8,0] = -0.2 #NTS2 -> PreI
b[7,8] = -0.45*x #NTS1 -> NTS2
b[8,7] = -0.15*x #NTS2 -> NTS1
# b[7,4] = 0.3 #NTS1 -> RampI
# b[8,4] = 0.2 #NTS2 -> RampI


c = np.zeros((num_drives, num_nrns))
# Pons
c[0,0] = 0.0
c[0,1] = 0.0
c[0,2] = 0.0
c[0,3] = 0.1
c[0,4] = 0.2
c[0,5] = 0.2

c[0,7] = 0.64
c[0,8] = 0.5

c[0,9] = 0.5*y

#Rtn/BotC
c[1,0] = 0.00
c[1,1] = 0.0
c[1,3] = 0.1

#PreBotC
c[2,0] = 0.025

b = b.tolist()
c = c.tolist()

params["description"] = \
    "# 0- PreI   # 1 - EarlyI  # 2 - PostI \
    # 3 - AugE  # 4 - RampI   # 5 - EarlyI2\
    # 6 - Relay # 7 - NTS1    # 8 - NTS2\
    # 9 - KF"

params["b"] = b

params["c"] = c

json.dump(params, open('rCPG_swCPG.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
