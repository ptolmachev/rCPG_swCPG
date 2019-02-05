import pickle
import numpy as np

params = dict()

num_nrns = 10
num_drives = 3
x = 1
y = 1

b = np.zeros((num_nrns, num_nrns))
# positive weights
b[0,1] = 0.6  #PreI -> EarlyI
b[0,4] = 1.5  #PreI -> RampI
b[9,0] = 0.4 # KF -> PreI
b[9,2] = 2.8 # KF -> PostI
b[9,3] = 0.4 # KF -> AugE
b[6,7] = 2.1 # Relay -> NTS1
b[6,8] = 1.6 # Relay -> NTS2
b[6,9] = 0.7 # Relay -> KF

# negative weights
b[1,2] = -0.3 #EarlyI -> PostI
b[1,3] = -0.39 #EarlyI -> AugE

b[1,9] = -0.05 #EarlyI -> KF #!!!

b[2,0] = -0.26 #PostI -> PreI
b[2,1] = -0.07 #PostI -> EarlyI
b[2,3] = -0.37 #PostI -> AugE
b[2,4] = -0.6  #PostI -> RampI
b[2,5] = -0.3  #PostI -> EarlyI2
b[3,4] = -0.2  #AugE -> RampI
b[3,0] = -0.4  #AugE -> PreI
b[3,1] = -0.3  #AugE -> EarlyI
b[3,2] = -0.1  #AugE -> PostI
b[5,4] = -0.3  #EarlyI2 -> RampI

b[7,0] = -0.3 #NTS1 -> PreI
b[8,0] = -0.2 #NTS2 -> PreI
b[7,8] = -0.45 #NTS1 -> NTS2
b[8,7] = -0.4*x #NTS2 -> NTS1
# b[7,4] = 0.3 #NTS1 -> RampI
# b[8,4] = 0.2 #NTS2 -> RampI

b[9,7] = -0.22*x #KF -> NTS1
b[9,8] = -0.19 #KF -> NTS2 #assymetric inhibition

c = np.zeros((num_drives, num_nrns))
# Pons
c[0,0] = 0.3
c[0,1] = 0.0
c[0,2] = 0.0
c[0,3] = 0.0
c[0,4] = 0.5
c[0,5] = 0.4

c[0,7] = 0.2
c[0,8] = 0.5
c[0,9] = 0.25*y

#Rtn/BotC
c[1,0] = 0.15
c[1,1] = 0.25
c[1,3] = 0.4

#PreBotC
c[2,0] = 0.025

params["b"] = b
params["c"] = c

with open('rCPG_swCPG.prms', 'wb+') as outfile:
    pickle.dump(params, outfile)