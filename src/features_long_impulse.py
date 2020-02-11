import numpy as np
from utils import *
from params_gen import *
import pickle
import json

if __name__ == '__main__':
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])
    t1_s = [10000, 14000, 18000] # 22000, 25000, 29000, 33000, 37000, 40000, 44000, 48000
    amps = [0 + i*3 for i in range(200)][::-1]
    periods = np.empty((len(amps), len(t1_s)), dtype = float)
    period_stds = np.empty((len(amps), len(t1_s)), dtype = float)
    rough_periods = np.empty((len(amps), len(t1_s)), dtype = float)
    num_swallows_s = np.empty((len(amps), len(t1_s)), dtype = int)
    num_breakthroughs_PreI_s = np.empty((len(amps), len(t1_s)), dtype = int)
    num_breakthroughs_AugE_s = np.empty((len(amps), len(t1_s)), dtype = int)
    for i in range(len(amps)):
        for j in range(len(t1_s)):
            amp = amps[i]
            t1 = t1_s[j]
            print("Amp: {}, time : {}".format(amp, t1))
            t2 = t1 + 10000
            stoptime = 60000
            signals, t = run_model(t1, t2, amp, stoptime, '10_sec_stim_diff_amp')
            # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
            period, period_std, rough_period, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE = \
                get_features_long_impulse(signals, t, t1, t2)
            periods[i,j] = period
            period_stds[i,j] = period_std
            rough_periods[i,j] = rough_period
            num_swallows_s[i,j] = num_swallows
            num_breakthroughs_AugE_s[i,j] = num_breakthroughs_AugE
            num_breakthroughs_PreI_s[i, j] = num_breakthroughs_PreI

    info = dict()
    info['amps'] = amps
    info['start_times'] = t1_s
    info['periods'] = periods
    info['period_stds'] = period_stds
    info['rough_periods'] = rough_periods
    info['num_swallows_s'] = num_swallows_s
    info['num_breakthroughs_AugE_s'] = num_breakthroughs_AugE_s
    info['num_breakthroughs_PreI_s'] = num_breakthroughs_PreI_s

    periods_avg = np.nanmean(periods, axis = 1)
    period_std_avg = np.nanmean(period_stds, axis=1)
    rough_periods_avg = np.nanmean(rough_periods, axis=1)
    num_swallows_s_avg = np.nanmean(num_swallows_s, axis=1)
    num_breakthroughs_AugE_s_avg = np.nanmean(num_breakthroughs_AugE_s, axis=1)
    num_breakthroughs_PreI_s_avg = np.nanmean(num_breakthroughs_PreI_s, axis=1)

    pickle.dump(info, open('features_var_amp.pkl', 'wb+'))
    nice_plot(periods_avg)
    nice_plot(period_std_avg)
    nice_plot(rough_periods_avg)
    nice_plot(num_swallows_s_avg)
    nice_plot(num_breakthroughs_AugE_s_avg)
    nice_plot(num_breakthroughs_PreI_s_avg)



