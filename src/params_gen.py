import json
import numpy as np

params = dict()

num_nrns = 15
num_drives = 3
x = [0.6,1.0,7.0][1] # Disinh-inh of NTS
y = [0.2, 1.0, 7.0][1] # Disinh-inh of KF

# 0- PreI   # 1 - EarlyI  # 2 - PostI
# 3 - AugE  # 4 - RampI   # 5 - Relay
# 6 - NTS1  # 7 - NTS2    # 8 - NTS3
# 9 - KF    # 10 - M_HN    # 11- M_PN
# 12 - M_VN # 13 - KF_inh # 14 - NTS_inh
b = np.zeros((num_nrns, num_nrns))
# positive weights
b[0,0] = 0.2  #PreI -> PreI
b[0,1] = 0.6  #PreI -> EarlyI
b[0,4] = 0.9  #PreI -> RampI
b[0,10] = 1.5 # PreI -> M_HN
b[2,12] = 3.1 # PostI -> M_VN
b[4,11] = 2.8 # RampI -> M_PN
b[4,12] = 1.5 # RampI -> M_VN
b[5,3] = 0.2 # Relay -> AugE
b[5,6] = 2.0 # Relay -> NTS1
b[5,7] = 2.0 # Relay -> NTS2
b[5,8] = 2.1 # Relay -> NTS3
b[5,9] = 1.5 # Relay -> KF #test
b[6,12] = 0.8 # NTS1 -> M_VN
b[6,10] = 0.7 # NTS1 -> H_VN
# b[7,3] = 0.25 # NTS2 -> AugE
b[8,2] = 1.15 # NTS3 -> PostI
b[8,3] = 0.22 # NTS3 -> AugE
b[8,6] = 0.3 # NTS3 -> NTS1
b[8,7] = 0.3 # NTS3 -> NTS2
b[9,2] = 0.5 # KF -> PostI
# b[9,3] = 0.1 # KF -> AugE
b[9,8] = 1.2 # KF -> NTS3

# negative weights
b[1,2] = -0.25 #EarlyI -> PostI
b[1,3] = -0.39 #EarlyI -> AugE
b[1,4] = -0.15  #EarlyI1 -> RampI
b[1,6] = -0.15  #EarlyI1 -> NTS1
b[1,7] = -0.15  #EarlyI1 -> NTS2
b[1,9] = -0.01  #EarlyI1 -> KF
b[2,0] = -0.1 #PostI -> PreI
b[2,1] = -0.1 #PostI -> EarlyI
b[2,3] = -0.57 #PostI -> AugE
b[2,4] = -0.4  #PostI -> RampI
b[3,4] = -0.2  #AugE -> RampI
b[3,0] = -0.2  #AugE -> PreI
b[3,1] = -0.3  #AugE -> EarlyI
b[3,2] = -0.54  #AugE -> PostI
b[6,2] = -0.06 #NTS1 -> PostI
b[7,2] = -0.05 #NTS2 -> PostI
b[6,7] = -0.3*x #NTS1 -> NTS2
# b[6,8] = -0.05*x #NTS1 -> NTS3
b[7,6] = -0.31*x #NTS2 -> NTS1
# b[7,8] = -0.05*x #NTS2 -> NTS3
b[9,6] = -0.08*x #KF -> NTS1
b[9,7] = -0.07*x #KF -> NTS2
b[13,9] = -0.3*y #KF_inh -> KF
b[14,5] = -0.3*x #NTS_inh -> Relay
b[14,6] = -0.2*x #NTS_inh -> NTS1
b[14,7] = -0.2*x #NTS_inh -> NTS2
b[14,8] = -0.2*x #NTS_inh -> NTS3

c = np.zeros((num_drives, num_nrns))
# other
c[0,0] = 0.0 #To PreI
c[0,1] = 0.0 #To EarlyI
c[0,2] = 0.07 #To PostI
c[0,3] = 0.1 #To AugE
c[0,4] = 0.0 #To RampI
c[0,6] = 0.8 #To NTS1
c[0,7] = 1.0 #To NTS2
c[0,8] = 0.1 #To NTS3
c[0,9] = 1.4 #To KF
c[0,13] = 0.5 #To KF_inh
c[0,14] = 0.4 #To NTS_inh
#Rtn/BotC
c[1,0] = 0.0 #To PreI
c[1,1] = 0.0 #To EarlyI
c[1,3] = 0.2 #To AugE

#PreBotC
c[2,0] = 0.025 #To PreI

b = b.tolist()
c = c.tolist()

params["description"] = ""

params["b"] = b

params["c"] = c

json.dump(params, open('rCPG_swCPG.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
