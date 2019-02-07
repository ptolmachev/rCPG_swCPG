import json
import numpy as np

params = dict()

num_nrns = 13
num_drives = 3
x = 1
y = 1.0 #1.1

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - EarlyI2
# 6 - Relay # 7 - NTS1    # 8 - NTS2
# 9 - KF #  # 10 - M_HN   # 11- M_PN
# 12 - M_VN

b = np.zeros((num_nrns, num_nrns))
# positive weights
b[0,1] = 0.6  #PreI -> EarlyI
b[0,4] = 1.5  #PreI -> RampI
b[9,0] = 0.4 # KF -> PreI
b[9,2] = 1.8 # KF -> PostI
b[9,3] = 0.5 # KF -> AugE
b[6,7] = 1.9 # Relay -> NTS1
b[6,8] = 1.9 # Relay -> NTS2
b[6,9] = 1.4 # Relay -> KF
b[6,2] = 0.4 # Relay -> PostI
b[0,10] = 1.5 # PreI -> M_HN
b[8,10] = 0.9 # NTS1 -> M_HN
b[4,11] = 2.8 # RampI -> M_PN
b[2,12] = 2.5 # PostI -> M_VN
b[4,12] = 0.9 # RampI -> M_VN
b[8,12] = 1.8 # NTS2 - M_VN
# negative weights
b[1,2] = -0.25 #EarlyI -> PostI
b[1,3] = -0.39 #EarlyI -> AugE

b[1,9] = -0.02 #EarlyI -> KF #!!!

b[2,0] = -0.26 #PostI -> PreI
b[2,1] = -0.07 #PostI -> EarlyI
b[2,3] = -0.37 #PostI -> AugE
b[2,4] = -0.6  #PostI -> RampI
b[2,5] = -0.3  #PostI -> EarlyI2
b[3,4] = -0.2  #AugE -> RampI
b[3,0] = -0.45  #AugE -> PreI
b[3,1] = -0.35  #AugE -> EarlyI
b[3,2] = -0.1  #AugE -> PostI
b[5,4] = -0.4  #EarlyI2 -> RampI

b[7,0] = -0.2 #NTS1 -> PreI
b[8,0] = -0.2 #NTS2 -> PreI
b[7,8] = -0.37 #NTS1 -> NTS2
b[8,7] = -0.425*x #NTS2 -> NTS1
# b[7,4] = 0.3 #NTS1 -> RampI
# b[8,4] = 0.2 #NTS2 -> RampI

b[9,7] = -0.22*x #KF -> NTS1
b[9,8] = -0.22 #KF -> NTS2 #assymetric inhibition

c = np.zeros((num_drives, num_nrns))
# Pons
c[0,0] = 0.2
c[0,1] = 0.0
c[0,2] = 0.0
c[0,3] = 0.0
c[0,4] = 0.6
c[0,5] = 0.2

c[0,7] = 0.64
c[0,8] = 0.5

c[0,9] = 0.5*y

#Rtn/BotC
c[1,0] = 0.15
c[1,1] = 0.25
c[1,3] = 0.3

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
