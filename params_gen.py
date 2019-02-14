import json
import numpy as np

params = dict()

num_nrns = 13
num_drives = 3
x = 1.0
y = [0.5,1.0,3.5][1] # Disinh-inh of KF

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - Relay
# 6 - NTS1  # 7 - NTS2    # 8 - KF
# 9 - M_HN  # 10- M_PN    # 11 - M_VN
# 12 - KF_inh
b = np.zeros((num_nrns, num_nrns))
# positive weights
b[0,1] = 0.6  #PreI -> EarlyI
b[0,4] = 0.9  #PreI -> RampI
b[0,9] = 1.5 # PreI -> M_HN
b[2,11] = 1.7 # PostI -> M_VN
b[4,10] = 2.8 # RampI -> M_PN
b[4,11] = 0.9 # RampI -> M_VN
b[5,6] = 0.2 # Relay -> NTS1
b[5,7] = 1.2 # Relay -> NTS2
b[5,8] = 2.5 # Relay -> KF
b[5,12] = 1.3 # Relay -> KF_inh
b[6,0] = 0.3 # NTS1 -> PreI
b[6,1] = 0.4 # NTS1 -> EarlyI1
b[6,3] = 0.1 # NTS1 -> AugE
b[7,2] = 0.1 # NTS2 -> PostI
b[7,3] = 0.3 # NTS2 -> AugE
b[7,9] = 0.9 # NTS2 -> M_HN
b[7,11] = 1.8 # NTS2 - M_VN
b[8,2] = 0.9 # KF -> PostI
b[8,3] = 0.15 # KF -> AugE
b[8,6]  =  0.1 #KF -> NTS1
b[8,7]  =  0.1 #KF -> NTS2

# negative weights
b[1,2] = -0.15 #EarlyI -> PostI
b[1,3] = -0.39 #EarlyI -> AugE
b[1,4] = -0.15  #EarlyI1 -> RampI
b[1,8] = -0.02 #EarlyI -> KF #!!!
b[2,0] = -0.13 #PostI -> PreI
b[2,1] = -0.1 #PostI -> EarlyI
b[2,3] = -0.4 #PostI -> AugE
b[2,4] = -0.4  #PostI -> RampI
b[3,4] = -0.2  #AugE -> RampI
b[3,0] = -0.27  #AugE -> PreI
b[3,1] = -0.3  #AugE -> EarlyI
b[3,2] = -0.45  #AugE -> PostI
b[6,7] = -0.15*x #NTS1 -> NTS2
b[7,6] = -0.15*x #NTS2 -> NTS1
b[12,8] = -0.2*y #KF_inh -> KF
# b[6,0] = -0.1 #NTS1 -> PreI
# b[7,0] = -0.2 #NTS2 -> PreI
# b[7,4] = 0.3 #NTS1 -> RampI
# b[8,4] = 0.2 #NTS2 -> RampI

c = np.zeros((num_drives, num_nrns))
# other
c[0,0] = 0.0 #To PreI
c[0,1] = 0.0 #To EarlyI
c[0,2] = 0.0 #To PostI
c[0,3] = 0.0 #To AugE
c[0,4] = 0.0 #To RampI
c[0,6] = 0.0 #To NTS1
c[0,7] = 0.0 #To NTS2
c[0,8] = 1.1 #To KF
c[0,12] = 0.4 #To KF_inh
#Rtn/BotC
c[1,0] = 0.0 #To PreI
c[1,1] = 0.0 #To EarlyI
c[1,3] = 0.2 #To AugE

#PreBotC
c[2,0] = 0.025 #To PreI

b = b.tolist()
c = c.tolist()

params["description"] = \
    "# 0- PreI   # 1 - EarlyI  # 2 - PostI \
    # 3 - AugE  # 4 - RampI \
    # 5 - Relay # 6 - NTS1    # 7 - NTS2\
    # 8 - KF"

params["b"] = b

params["c"] = c

json.dump(params, open('rCPG_swCPG.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
