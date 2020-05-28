import json
import numpy as np
from utils.gen_utils import get_project_root

def generate_params(inh_NTS, inh_KF):
    params = dict()
    num_nrns = 18
    num_drives = 3
    x = [0.1, 1.0, 10][inh_NTS] # Disinh-inh of NTS
    y = [0.1, 1.0, 10][inh_KF] # Disinh-inh of KF
    # 0- PreI   # 1 - EarlyI  # 2 - PostI
    # 3 - AugE  # 4 - RampI   # 5 - Relay
    # 6 - Sw 1  # 7 - Sw2     # 8 - Sw3
    # 9 - KF_t   # 10 - KF_p   # 11 - KF_r
    # 12 - M_HN  # 13- M_PN  # 14 - M_VN
    # 15 - KF_inh # 16 - NTS_inh # 17 - SI
    b = np.zeros((num_nrns, num_nrns))
    # # positive weights
    b[0,1] = 0.3  #PreI -> EarlyI # Rubins (2009): (0.4)    Rubins (2011): (0.35)
    b[0,4] = 0.6  #PreI -> RampI
    b[0,12] = 0.4 # PreI -> M_HN
    b[2,14] = 0.25 # PostI -> M_VN
    b[4,13] = 0.6 # RampI -> M_HN
    b[4,13] = 0.85 # RampI -> M_PN
    b[4,14] = 0.75 # RampI -> M_VN
    # b[5,2] = 0.0 # Relay -> PostI
    b[5,6] = 0.69 # Relay -> Sw1
    b[5,7] = 0.75 # Relay -> Sw2
    b[5,8] = 0.1 # Relay -> Sw3
    b[5,9] = 0.4 # Relay -> KF_t
    b[5,10] = 0.3 # Relay -> KF_p
    b[6,2] = 0.2 # Sw1 -> PostI
    b[6,12] = 0.5 # Sw1 -> M_HN
    b[6,14] = 0.6 # Sw1 -> M_VN
    b[8,1] = 0.2 # Sw3 -> EarlyI
    b[8,2] = 0.5 # Sw3 -> PostI
    b[10,2] = 0.6 # KF_p -> PostI
    b[10,8] = 0.5 # KF_p -> Sw3
    b[10,14] = 0.38 # KF_p -> M_VN
    b[9, 11] = 1.1  # KF_t -> KF_relay
    b[17, 5] = 0.75  # SI -> Relay

    # negative weights
    # negative weights
    b[1, 0] = -0.02  # EarlyI -> PreI #in Rubins: (0)    Rubins (2011): (0)
    b[1, 2] = -0.3  # EarlyI -> PostI #in Rubins: (0.25)    Rubins (2011): (0.2)
    b[1, 3] = -0.45  # EarlyI -> AugE #in Rubins: (0.35)    Rubins (2011): (0.25)
    b[1, 4] = -0.25  # EarlyI1 -> RampI
    b[1, 10] = -0.3  # EarlyI1 -> KF_p

    b[2, 0] = -0.16  # PostI -> PreI #in Rubins: (0.3)    Rubins (2011): (0.8)
    b[2, 1] = -0.38  # PostI -> EarlyI #in Rubins: (0.05)    Rubins (2011): (0.15)
    b[2, 3] = -0.35  # PostI -> AugE #in Rubins: (0.35)    Rubins (2011): (0.4)
    b[2, 4] = -0.67  # PostI -> RampI

    b[3, 0] = -0.55  # 0.55 AugE -> PreI #in Rubins: (0.2)    Rubins (2011): (0.22)
    b[3, 1] = -0.42  # AugE -> EarlyI #in Rubins: (0.35)    Rubins (2011): (0.08)
    b[3, 2] = -0.02  # AugE -> PostI #in Rubins: (0.1)    Rubins (2011): (0.0)
    b[3, 4] = -0.67  # AugE -> RampI

    b[5, 0] = -0.2  # Relay -> PreI
    b[5, 1] = -0.2  # Relay -> EarlyI

    b[6, 0] = -0.2  # Sw1 -> PreI
    b[6, 1] = -0.2  # Sw1 -> EarlyI
    b[6, 7] = -0.55 * x  # Sw1 -> Sw2
    b[7, 6] = -0.39 * x  # Sw2 -> Sw1

    b[11, 0] = -0.1  # KF_relay -> PreI
    b[11, 1] = -0.1  # KF_relay -> EarlyI
    b[11, 6] = -0.05  # KF_relay -> Sw1
    b[11, 7] = -0.05  # KF_relay -> Sw2

    b[15, 9] = -0.3 * y  # KF_inh -> KF_t
    b[15, 10] = -0.3 * y  # KF_inh -> KF_p
    b[16, 5] = -0.3 * x  # NTS_inh -> Relay
    b[16, 6] = -0.1 * x  # NTS_inh -> Sw1
    b[16, 7] = -0.1 * x  # NTS_inh -> Sw2
    b[16, 8] = -0.1 * x  # NTS_inh -> Sw3

    c = np.zeros((num_drives, num_nrns))
    # other
    c[0,0] = 0.262 #To PreI
    c[0,1] = 0.39  #To EarlyI
    c[0,2] = 0.03  #To PostI
    c[0,3] = 0.385  #To AugE
    c[0,4] = 0.53  #To RampI
    c[0,6] = 0.34 #To Sw1
    c[0,7] = 0.45  #To Sw2
    c[0,8] = 0.3  #To Sw3
    c[0,9] = 0.8  #To KF_t
    c[0,10] = 0.8  #To KF_p
    c[0,15] = 0.3 #To KF_inh
    c[0,16] = 0.3 #To NTS_inh

    b = b.tolist()
    c = c.tolist()

    params["description"] = ""
    params["b"] = b
    params["c"] = c
    data_path = str(get_project_root()) + "/data"
    json.dump(params, open(f'{data_path}/rCPG_swCPG.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    return None

if __name__ == '__main__':
    #NTS, KF
    generate_params(1, 1)