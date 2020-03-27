import pickle
import numpy as np
from matplotlib import pyplot as plt
from utils.sp_utils import get_onsets_and_ends


def clarifying_plot(chunk, save_to):
    PNA = chunk['signal']
    s = int(0.4 * len(PNA))
    e = int(0.8 * len(PNA))
    stim = chunk['stim'] - s
    PNA = (PNA[s:e])
    insp_begins, insp_ends = get_onsets_and_ends(PNA, model='l2', pen=1000, min_len=100)
    ts1, ts2, ts3, ts4, te1, te2, te3, te4 = get_onsets_and_ends(insp_begins, insp_ends, stim,  min_len=100)
    PNA = (PNA - np.min(PNA)) / (np.max(PNA) - np.min(PNA))

    class DoubleArrow():
        def __init__(self, pos1, pos2, level, margin):
            plt.arrow(pos1 + margin, level, pos2 - pos1 - 2 * margin, 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k')
            plt.arrow(pos2, level, pos1 - pos2 + 2 * margin , 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k', head_starts_at_zero = True)

    class ConvergingArrows():
        def __init__(self, pos1, pos2, level, margin):
            plt.arrow(pos1+10 - 10*margin, level, 8*margin, 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k')
            plt.arrow(pos2 + 10*margin, level, -8* margin , 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k', head_starts_at_zero = True)


    fig = plt.figure(figsize=(20, 6))
    plt.plot(PNA, linewidth=2, color='k')
    margin = 5
    height0 = 0.9
    height1 = 1.05
    height2 = 0.00
    height3 = -0.05
    stim_duration = 75
    ts1 = ts1-40
    ts2 = ts2 - 20
    ConvergingArrows(stim, stim+stim_duration, height0, margin)  # Stim
    DoubleArrow(ts1, ts2, height1, margin) # T0
    DoubleArrow(ts2, ts3, height1, margin)  # T1
    DoubleArrow(ts1, te1, height2, margin)  # Ti_0
    DoubleArrow(ts3, te3, height2, margin)  # Ti_1
    DoubleArrow(ts4, te4, height2, margin)  # Ti_2
    DoubleArrow(ts2, stim, height3, margin)  # Phi
    DoubleArrow(stim+stim_duration, ts3, height3, margin)  # Theta

    plt.axvline(stim, color='r', linestyle='--')
    plt.axvline(stim + stim_duration, color='r', linestyle='--')
    plt.axvspan(stim, stim + stim_duration, color='r', alpha=0.3)
    plt.axvline(ts1, color='b', linestyle='--')
    plt.axvline(ts2, color='b', linestyle='--')
    plt.axvline(ts3, color='b', linestyle='--')
    plt.axvline(ts4, color='b', linestyle='--')
    plt.axvline(te1, color='b', linestyle='--')
    plt.axvline(te2, color='b', linestyle='--')
    plt.axvline(te3, color='b', linestyle='--')
    plt.axvline(te4, color='b', linestyle='--')

    plt.title("Phrenic Nerve Activity", fontsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.1, 1.1])
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    plt.axis('off')
    plt.savefig(f"{save_to}")
    plt.close()
    return None

if __name__ == '__main__':
    # # PLOT WITH MEANING OF THE PARAMETERS
    num_rec = 3
    num_chunk = 6
    save_to = f'../../img/param_representation.png'
    data = pickle.load(open(f'../../data/sln_prc_chunked/2019-09-05_12-26-14_prc/100_CH10_chunked.pkl', 'rb+'))
    chunk = data[num_chunk]
    clarifying_plot(chunk, save_to)
