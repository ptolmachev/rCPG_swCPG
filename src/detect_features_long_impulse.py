# this script contains count of swallowing peaks, instantaneous frequency, breakthrough delay and the respiratory delay for various amplitude signals
import numpy as np
from plot_signals import plot_signals
from model import *
import json
from scipy import signal
from scipy.integrate import odeint
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sg
import pickle

def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    plt.plot(series, 'r-',linewidth = 2, alpha = 0.7)
    plt.show()

def get_features_long_impulse(signals,t, t1, t2):
    #first one has to cut the relevant signal:
    labels = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF", "Motor_HN", "Motor_PN",
              "Motor_VN", "KF_inh", "NTS_inh"]
    needed_labels = ["PreI", "PostI", "AugE", "NTS1"]
    ind1 = np.where(np.array(t) <= t1)[0][-1]
    ind2 = np.where(np.array(t) <= t2)[0][-1]
    t = np.array(t[ind1:ind2]) - t[ind1]
    signals_relevant = [signals[i][ind1:ind2] for i in range(len(signals)) if labels[i] in needed_labels]
    filename = "test"

    NTS1 = signals_relevant[needed_labels.index("NTS1")]
    PreI = signals_relevant[needed_labels.index("PreI")]
    PostI = signals_relevant[needed_labels.index("PostI")]
    AugE = signals_relevant[needed_labels.index("AugE")]

    #identifying instantaneous frequency:
    corr = sg(scipy.signal.correlate(NTS1, NTS1,'same'),121,1)
    peaks = scipy.signal.find_peaks(corr)[0]

    if len(peaks) >= 3:
        period = np.mean([peaks[i] - peaks[i-1] for i in range(1,len(peaks))])*(t2-t1)/len(t)
        period_std = np.std([peaks[i] - peaks[i-1] for i in range(1,len(peaks))])*(t2-t1)/len(t)
    else:
        period = np.inf
        period_std = 0

    # print("period: {} +- {}".format(period, period_std))

    #identifying the number of swallows in total
    swallows = [swallow_id for swallow_id in (scipy.signal.find_peaks(NTS1)[0]) if (NTS1[swallow_id] > 0.15) and (swallow_id > 50)]
    num_swallows = len(swallows)
    # print("num_swallows: {}".format(num_swallows))

    #identifying the number of PreI breakthroughs:
    breakthroughs_PreI = [breakthrough_id for breakthrough_id in (scipy.signal.find_peaks(PreI)[0]) if PreI[breakthrough_id] > 0.15]
    num_breakthroughs_PreI  = len(breakthroughs_PreI)
    # print("num_breakthroughs_PreI: {}".format(num_breakthroughs_PreI))

    #identifying the number of AugE breakthroughs:
    breakthroughs_AugE = [breakthrough_id for breakthrough_id in (scipy.signal.find_peaks(AugE)[0]) if AugE[breakthrough_id] > 0.15]
    num_breakthroughs_AugE  = len(breakthroughs_AugE)
    # print("num_breakthroughs_Aug: {}".format(num_breakthroughs_AugE))

    #Rough period estimation:
    if num_swallows != 0:
        period_rough = 10000/num_swallows
    else:
        period_rough = np.inf
    # print("Rough period estimation: {}".format(period_rough))

    # plot_signals(t, signals_relevant, needed_labels, 0, t[-1], filename)
    return period, period_std, period_rough, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE



if __name__ == '__main__':
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])
    t1_s = [8000, 10000,17500, 20000, 25000, 27500, 30500, 35000, 37500,25000 + 100*np.random.randn(), 30500 + 100*np.random.randn()]
    amps = [100 + i*3 for i in range(120)]
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
            res = model(b, c, vectorfield, t1, t2, amp, stoptime)
            t = res[0]
            signals = res[1:]
            period, period_std, rough_period, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE = get_features_long_impulse(signals,t, t1, t2)
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

    nice_plot(periods_avg)
    nice_plot(period_std_avg)
    nice_plot(rough_periods_avg)
    nice_plot(num_swallows_s_avg)
    nice_plot(num_breakthroughs_AugE_s_avg)
    nice_plot(num_breakthroughs_PreI_s_avg)

    pickle.dump(info, open('features_var_amp_2.pkl', 'wb+'))


