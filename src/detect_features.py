# this script contains count of swallowing peaks, instantaneous frequency, breakthrough delay and the respiratory delay for various amplitude signals
import numpy as np
from plot_signals import plot_signals
from model import *
import json
from scipy.integrate import odeint
import scipy

def get_features(signals,t, t1, t2):
    #first one has to cut the relevant signal:

    ind1 = np.where(np.array(t) <= 25000)[0][-1]
    ind2 = np.where(np.array(t) <= 35000)[0][-1]
    t = np.array(t[ind1:ind2]) - t[ind1]
    signals_relevant = [signal[ind1:ind2] for signal in signals]
    filename = "test"
    labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF","Motor_HN", "Motor_PN", "Motor_VN","KF_inh", "NTS_inh"]
    plot_signals(t, signals_relevant, labels, t[-1], filename)


if __name__ == '__main__':
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])
    t1 = 25000
    t2 = 35000
    amp = 250
    stoptime = 60000
    res = model(b, c, vectorfield, t1, t2, amp, stoptime)
    t = res[0]
    signals = res[1:]
    get_features(signals,t, t1, t2)