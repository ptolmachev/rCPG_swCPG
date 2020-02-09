import json
import numpy as np


def generate_params(inh_NTS, inh_KF):
    params = dict()

    num_nrns = 16
    num_drives = 3
    x = [0.1, 1.0, 10][inh_NTS] # Disinh-inh of NTS
    y = [0.1, 1.0, 10][inh_KF] # Disinh-inh of KF

    # 0- PreI   # 1 - EarlyI  # 2 - PostI
    # 3 - AugE  # 4 - RampI   # 5 - Relay
    # 6 - Sw 1  # 7 - Sw2     # 8 - Sw3
    # 9 - KFi   # 10 - KFe    # 11 - M_HN
    # 12- M_PN  # 13 - M_VN   # 14 - KF_inh
    # 15 - NTS_inh
    b = np.zeros((num_nrns, num_nrns))
    # positive weights
    b[0,0] = 0.00  #PreI -> PreI
    b[0,1] = 0.3  #PreI -> EarlyI
    b[0,4] = 0.6  #PreI -> RampI
    b[0,12] = 0.4 # PreI -> M_PN
    b[0,11] = 0.4 # PreI -> M_HN
    b[2,13] = 0.3 # PostI -> M_VN
    b[4,12] = 0.6 # RampI -> M_HN
    b[4,12] = 0.4 # RampI -> M_PN
    b[4,13] = 0.6 # RampI -> M_VN
    b[5,2] = 0.4 # Relay -> PostI
    b[5,6] = 0.75 # Relay -> Sw1
    b[5,7] = 0.75 # Relay -> Sw2
    b[5,8] = 0.65 # Relay -> Sw3
    b[5,9] = 0.4 # Relay -> KFi
    b[5,10] = 0.4 # Relay -> KFe
    b[6,11] = 0.5 # NTS1 -> M_HN
    b[6,13] = 0.45 # NTS1 -> M_VN
    b[8,1] = 0.2 # NTS3 -> EarlyI
    b[8,2] = 0.4 # NTS3 -> PostI
    b[10,0] = 0.1 # KFe -> PreI
    b[10,1] = 0.1 # KFe -> EarlyI
    b[10,2] = 0.85 # KFe -> PostI
    b[10,8] = 0.5 # KFe -> NTS3
    b[10,13] = 0.5 # KFe -> M_VN

    # negative weights
    b[1,2] = -0.3   #EarlyI -> PostI
    b[1,3] = -0.4  #EarlyI -> AugE
    b[1,6] = -0.00005  #EarlyI -> Sw1
    b[1,7] = -0.00005  #EarlyI -> Sw2
    b[1,4] = -0.08  #EarlyI1 -> RampI
    b[1,10] = -0.3  #EarlyI1 -> KFe

    b[2,0] = -0.18    #PostI -> PreI
    b[2,1] = -0.28   #PostI -> EarlyI
    b[2,3] = -0.35   #PostI -> AugE
    b[2,4] = -0.2    #PostI -> RampI
    b[2,6] = -0.01  #PostI -> Sw1
    b[2,7] = -0.01  #PostI -> Sw2

    b[3,4] = -0.25  #AugE -> RampI
    b[3,0] = -0.56   #AugE -> PreI
    b[3,1] = -0.46  #AugE -> EarlyI
    b[3,2] = -0.04  #AugE -> PostI
    b[3,6] = -0.05 #AugE -> Sw1
    b[3,7] = -0.05 #AugE -> Sw2
    b[3,9] = -0.01  #AugE -> KFi
    b[3,10] = -0.01 #AugE -> KFe
    b[3,13] = -0.1  #AugE -> M_VN

    b[5,0] = -0.2 # Relay -> PreI
    b[5,1] = -0.2 # Relay -> EarlyI

    # b[6,0] = -0.4 #Sw1 -> PreI
    b[6,7] = -0.3*x #Sw1 -> Sw2
    b[7,6] = -0.34*x #Sw2 -> Sw1

    b[9,0] = -0.04 #KFi -> PreI
    b[9,1] = -0.04 #KFi -> EarlyI
    b[9,6] = -0.025 #KFi -> Sw1
    b[9,7] = -0.018 #KFi -> Sw2

    b[14,9] = -0.3*y #KF_inh -> KFi
    b[14,10] = -0.3*y #KF_inh -> KFe
    b[15,5] = -0.3*x #NTS_inh -> Relay
    b[15,6] = -0.2*x #NTS_inh -> Sw1
    b[15,7] = -0.2*x #NTS_inh -> Sw2
    b[15,8] = -0.2*x #NTS_inh -> Sw3

    c = np.zeros((num_drives, num_nrns))
    # other
    c[0,0] = 0.2+0.01 #To PreI
    c[0,1] = 0.34+0.06  #To EarlyI
    c[0,2] = 0.03  #To PostI
    c[0,3] = 0.39  #To AugE
    c[0,4] = 0.3  #To RampI
    c[0,6] = 0.595 #To Sw1
    c[0,7] = 0.7  #To Sw2
    c[0,8] = 0.8  #To Sw3
    c[0,9] = 0.8  #To KFi
    c[0,10] = 0.8  #To KFe
    c[0,14] = 0.3 #To KF_inh
    c[0,15] = 0.3 #To NTS_inh

    #PreBotC
    c[2,0] = 0.025 #To PreI

    b = b.tolist()
    c = c.tolist()

    params["description"] = ""

    params["b"] = b

    params["c"] = c

    json.dump(params, open('rCPG_swCPG.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    return None
