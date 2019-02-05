from model import *
from plot_signals import plot_signals
import numpy as np

# DEFINING PARAMETERS
num_nrns = 10
num_drives = 3
b = np.zeros((num_nrns, num_nrns))
b[0,1] = 0.4
b[1, 2] = -0.25
b[1, 3] = -0.35
b[2, 0] = -0.3
b[2, 1] = -0.05
b[2, 3] = -0.35
b[3, 0] = -0.2
b[3, 1] = -0.35
b[3, 2] = -0.0

c = np.zeros((num_drives, num_nrns))
c[0, 0] = 0.115
c[0, 1] = 0.3
c[0, 2] = 0.63
c[0, 3] = 0.35
c[1, 0] = 0.07
c[1, 1] = 0.3
c[1, 3] = 0.4
c[2, 0] = 0.025

res = model(b, c, vectorfield)
t = res[0]
signals = res[1:]
plot_signals(t,signals)