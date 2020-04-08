from num_experiments.run_model import run_model
from src.utils.sp_utils import *
from src.num_experiments.params_gen import *
import pickle
from tqdm.auto import tqdm
import os

from utils.gen_utils import get_project_root


def get_features_long_impulse(signals, dt, t_stim_start, t_stim_finish):
    #first one has to cut the relevant signal:
    labels = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p", "KF_r",
              "Motor_HN", "Motor_PN", "Motor_VN", "KF_inh", "NTS_inh"]

    needed_labels = ["PreI", "AugE", "Sw1"]
    ind_stim_start = int(t_stim_start / dt) + 10 # +10 for transients
    ind_stim_finish = int(t_stim_finish / dt)
    signals_relevant = [signals[i][ind_stim_start:ind_stim_finish] for i in range(len(signals)) if labels[i] in needed_labels]

    Sw1 = signals_relevant[needed_labels.index("Sw1")]
    PreI = signals_relevant[needed_labels.index("PreI")]
    AugE = signals_relevant[needed_labels.index("AugE")]

    period, period_std = get_period(Sw1)

    #identifying the number of breakthroughs
    num_swallows = get_number_of_breakthroughs(Sw1, min_amp=0.2)
    num_breakthroughs_PreI = get_number_of_breakthroughs(PreI, min_amp=0.4)
    num_breakthroughs_AugE = get_number_of_breakthroughs(AugE, min_amp=0.1)

    #Rough period estimation:
    if num_swallows != 0:
        period_rough = (t_stim_finish - t_stim_start) / num_swallows
    else:
        period_rough = np.nan
    return period, period_std, period_rough, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE

def exctract_data(data_path, save_to):
    file_names = os.listdir(data_path)
    info = dict()
    for file_name in file_names:
        file = open(data_path + "/" + file_name, "rb+")
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
    pickle.dump(info, open(f"{save_to}/info_var_amp.pkl","wb+"))
    return None

# run all the simulations and save them into the folder
def run_simulations(dt, stim_duration, t_starts, stoptime, amps, save_to):
    generate_params(1, 1)
    for i in tqdm(range(len(amps))):
        for j in range(len(t_starts)):
            amp = amps[i]
            t_start = t_starts[j]
            # print("Amp: {}, time : {}".format(amp, t1))
            t_end = t_start + stim_duration
            signals, t = run_model(dt, t_start, t_end, amp, stoptime)
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['amp'] = amp
            data['start_stim'] = t_start
            data['duration'] = stim_duration
            pickle.dump(data, open(f"{save_to}/run_{amp}_{t_start}_{stim_duration}.pkl", "wb+"))
    return None

if __name__ == "__main__":
    data_path = str(get_project_root()) + "/data"
    # dt = 0.75
    # stoptime = 50000
    # t_starts = [22000, 25000, 29000, 33000, 37000]
    # stim_duration = 10000
    # amps = [100 + i*3 for i in range(150)][::-1]
    # save_to = data_path + "/" + "num_exp_runs" "/" + "long_stim"
    # run_simulations(dt, stim_duration, t_starts, stoptime, amps, save_to)

    data_folder = data_path + "/" + "num_exp_runs" + "/" + "long_stim"
    save_to = data_path + "/" + "num_exp_results" + "/" + "long_stim"
    exctract_data(data_folder, save_to)

