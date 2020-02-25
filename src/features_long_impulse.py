import numpy as np
from utils import *
from params_gen import *
import pickle
import json
from tqdm.auto import tqdm
import os
from matplotlib import pyplot as plt
from params_gen import generate_params


def exctract_data(path):
    file_names = os.listdir(path)
    info = dict()
    for file_name in file_names:
        file = open(path + "/" + file_name, "rb+")
        data = pickle.load(file)
        signals = data['signals']
        t = data['t']
        dt = data['dt']
        amp = data['amp']
        t1 = data['start_stim']
        stim_duration = data['duration']
        # if info key already exists, then pass
        if not (amp in list(info.keys())):
            info[amp] = dict()
            info[amp]["sw_period"] = []
            info[amp]["sw_period_std"] = []
            info[amp]["rough_period"] = []
            info[amp]["num_swallows"] = []
            info[amp]["num_breakthroughs_PreI"] = []
            info[amp]["num_breakthroughs_AugE"] = []

        sw_period, period_std, rough_period, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE = \
            get_features_long_impulse(signals, dt, t1, t1 + stim_duration)

        info[amp]["sw_period"].append(sw_period)
        info[amp]["sw_period_std"].append(period_std)
        info[amp]["rough_period"].append(rough_period)
        info[amp]["num_swallows"].append(num_swallows)
        info[amp]["num_breakthroughs_PreI"].append(num_breakthroughs_PreI)
        info[amp]["num_breakthroughs_AugE"].append(num_breakthroughs_AugE)
        file.close()

    pickle.dump(info, open("../data/info_var_amp.pkl","wb+"))
    return None

# def process_data(path):
#     info = pickle.load(open(path, "rb+"))
#     amps = list(info.keys())
#     amps.sort()
#     data = np.zeros((len(amps), 6))
#     for i, amp in tqdm(enumerate(amps)):
#         data[i, 0] = np.nanmean(info[amp]["sw_period"])
#         data[i, 1] = np.nanmean(info[amp]["sw_period_std"])
#         data[i, 2] = np.nanmean(info[amp]["rough_period"])
#         data[i, 3] = np.nanmean(info[amp]["num_swallows"])
#         data[i, 4] = np.nanmean(info[amp]["num_breakthroughs_PreI"])
#         data[i, 5] = np.nanmean(info[amp]["num_breakthroughs_AugE"])

# run all the simulations and save them into the folder
def run_simulations():
    generate_params(1, 1)
    t1_s = [22000, 25000, 29000, 33000, 37000]
    stim_duration = 10000
    amps = [100 + i*3 for i in range(150)][::-1]
    for i in tqdm(range(len(amps))):
        for j in range(len(t1_s)):
            amp = amps[i]
            t1 = t1_s[j]
            # print("Amp: {}, time : {}".format(amp, t1))
            t2 = t1 + stim_duration
            stoptime = 50000
            signals, t = run_model(t1, t2, amp, stoptime, '10_sec_stim_diff_amp')
            dt = 0.75
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['amp'] = amp
            data['start_stim'] = t1
            data['duration'] = stim_duration
            pickle.dump(data, open(f"../data/long_stim/run_{amp}_{t1}.pkl", "wb+"))
    return None

if __name__ == "__main__":
    # run_simulations()
    # exctract_data("../data/long_stim")
    process_data("../data/info_var_amp.pkl")
