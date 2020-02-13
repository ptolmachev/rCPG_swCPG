import pickle
import json
import numpy as np
from utils import *
from tqdm.auto import tqdm

file = open("../data/rCPG_swCPG.json", "rb+")
params = json.load(file)
b = np.array(params["b"])
c = np.array(params["c"])
# first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
t1 = 0
t2 = 100
stoptime = 70000
# amp = 0
# signals, t = run_model(t1, t2, amp, stoptime, '100ms_stim_diff_phase')
# pickle.dump((signals, t), open("../data/signals_intact_model.pkl", "wb+"))
signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
# get rid of transients 20000:
# warning period is in indices not in ms!
T, T_std = get_period(signals[:, 20000:])

amp = 370
# start from the end of expiration (begin of inspiration)
t_start_insp = (get_insp_starts(signals[:, 20000:]) + 20000) * t[0]
t1_s = t_start_insp[:9]
# shifts in ms
shifts = np.array([T * i / 10 for i in range(10)]) * t[0]  # dt

for i in tqdm(range(len(shifts))):
    shift = shifts[i]
    t1 = t1_s[-1] + shift
    # print("Shift: {}, Impulse at time : {}".format(shift, t1))
    t2 = t1 + 100
    stoptime = 70000
    # create and run a model
    signals, t = run_model(t1, t2, amp, stoptime, '100ms_stim_diff_phase')
    pickle.dump((signals, t), open(f"../data/signals_intact_model_{int(shift)}.pkl", "wb+"))
    # signals, t = pickle.load(open(f"../data/signals_intact_model_{int(shift)}.pkl", "rb+"))
    Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2 = get_features_short_impulse(signals, t, t1, t2)

