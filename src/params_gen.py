import json
import numpy as np

params = dict()

num_nrns = 15
num_drives = 3
x = [0.1, 1.0, 6][2] # Disinh-inh of NTS
y = [0.1, 1.0, 6][1] # Disinh-inh of KF

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - Relay
# 6 - NTS1  # 7 - NTS2    # 8 - NTS3
# 9 - KF    # 10 - M_HN    # 11- M_PN
# 12 - M_VN # 13 - KF_inh # 14 - NTS_inh
b = np.zeros((num_nrns, num_nrns))
# positive weights
b[0,0] = 0.00  #PreI -> PreI
b[0,1] = 0.3  #PreI -> EarlyI
b[0,4] = 0.6  #PreI -> RampI
b[0,11] = 0.3 # PreI -> M_PN
b[0,10] = 0.3 # PreI -> M_HN
b[2,12] = 0.3 # PostI -> M_VN
b[4,11] = 0.3 # RampI -> M_PN
b[4,12] = 0.3 # RampI -> M_VN
b[5,2] = 0.4 # Relay -> PostI
b[5,3] = 0.3 # Relay -> AugE
b[5,6] = 0.3 # Relay -> NTS1
b[5,7] = 0.3 # Relay -> NTS2
b[5,8] = 0.65 # Relay -> NTS3
b[5,9] = 0.4 # Relay -> KF
b[6,10] = 0.3 # NTS1 -> M_HN
b[6,12] = 0.3 # NTS1 -> M_VN
b[8,1] = 0.6 # NTS3 -> EarlyI
b[8,2] = 0.55 # NTS3 -> PostI
b[9,1] = 0.1 # KF -> EarlyI
b[9,2] = 0.7 # KF -> PostI
b[9,3] = 0.3 # KF -> AugE
b[9,7] = 0.3 # KF -> NTS2
b[9,8] = 0.5 # KF -> NTS3

factor1=0.4
# negative weights
b[1,2] = -0.3   #EarlyI -> PostI
b[1,3] = -0.25  #EarlyI -> AugE
b[1,6] = -0.05  #EarlyI -> NTS1
b[1,7] = -0.05  #EarlyI -> NTS2
b[1,4] = -0.08  #EarlyI1 -> RampI
b[1,9] = -0.05  #EarlyI1 -> KF

b[2,0] = -0.1    #PostI -> PreI
b[2,1] = -0.4    #PostI -> EarlyI
b[2,3] = -0.25   #PostI -> AugE
b[2,4] = -0.2    #PostI -> RampI
b[2,6] = -0.005  #PostI -> NTS1
b[2,7] = -0.005  #PostI -> NTS2


b[3,4] = -0.25  #AugE -> RampI
b[3,0] = -0.25  #AugE -> PreI
b[3,1] = -0.25  #AugE -> EarlyI
b[3,2] = -0.05   #AugE -> PostI
b[3,6] = -0.0   #AugE -> NTS1
b[3,7] = -0.0   #AugE -> NTS2
b[3,9] = -0.01  #AugE -> KF

b[5,0] = -0.2 # Relay -> PreI
b[5,1] = -0.2 # Relay -> EarlyI

b[6,7] = -0.3*x #NTS1 -> NTS2
b[7,6] = -0.2*x #NTS2 -> NTS1

# b[6,7] = -0.27*x #NTS1 -> NTS2
# b[7,6] = -0.235*x #NTS2 -> NTS1
# b[7,8] = -0.05*x #NTS2 -> NTS3

b[9,6] = -0.025 #KF -> NTS1
b[9,7] = -0.018 #KF -> NTS2

b[13,9] = -0.3*y #KF_inh -> KF
b[14,5] = -0.3*x #NTS_inh -> Relay
b[14,6] = -0.2*x #NTS_inh -> NTS1
b[14,7] = -0.2*x #NTS_inh -> NTS2
b[14,8] = -0.2*x #NTS_inh -> NTS2

c = np.zeros((num_drives, num_nrns))
# other
c[0,0] = 0.075 #To PreI
c[0,1] = 0.3  #To EarlyI
c[0,2] = 0.2  #To PostI
c[0,3] = 0.35  #To AugE
c[0,4] = 0.3  #To RampI
c[0,6] = 0.3 #To NTS1
c[0,7] = 0.3 #To NTS2
c[0,8] = 0.8  #To NTS3
c[0,9] = 0.8  #To KF
c[0,13] = 0.3 #To KF_inh
c[0,14] = 0.3 #To NTS_inh


#PreBotC
c[2,0] = 0.025 #To PreI

b = b.tolist()
c = c.tolist()

params["description"] = ""

params["b"] = b

params["c"] = c

json.dump(params, open('rCPG_swCPG.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
