import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
# rc('text', usetex=True)

def nice_error_bar(x,y,error, title, xlabel,ylabel, save_to = None):
    fig = plt.figure(figsize = (40,20))
    plt.errorbar(x,y, yerr = error , color = 'red',
             ecolor = 'gray', capsize = 3,linestyle = 'dashed', linewidth = 3, alpha = 0.8)

    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.ylim(min(y-error)-0.2*abs(min(y-error)),1.2*max(y+error))
    plt.grid(True)
    plt.show()
    if save_to is not None:
        fig.savefig("../img/" + save_to)


def nice_error_bar_scatter(x,y,error, title, xlabel,ylabel, save_to = None):

    fig = plt.figure(figsize = (40,20))
    plt.errorbar(x, y, yerr = error , color = 'red', ls='', marker='o',
             ecolor = 'gray', capsize = 3, linewidth = 3, alpha = 0.8)

    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.ylim(min(y-error)-0.2*abs(min(y-error)),1.2*max(y+error))
    plt.grid(True)
    plt.show()
    if save_to is not None:
        fig.savefig("../img/" + save_to)


if __name__ == '__main__':
    # start_idx = 0
    # info = pickle.load(open('../data/info_var_amp.pkl', "rb+"))
    # amps = list(info.keys())
    # amps.sort()
    # data = np.zeros((len(amps), 8))
    # for i, amp in (enumerate(amps)):
    #     data[i, 0] = np.nanmean(info[amp]["sw_period"])
    #     data[i, 1] = np.nanmean(info[amp]["sw_period_std"])
    #     data[i, 2] = np.nanmean(info[amp]["num_swallows"])
    #     data[i, 3] = np.std(info[amp]["num_swallows"])
    #     data[i, 4] = np.nanmean(info[amp]["num_breakthroughs_PreI"])
    #     data[i, 5] = np.std(info[amp]["num_breakthroughs_PreI"])
    #     data[i, 6] = np.nanmean(info[amp]["num_breakthroughs_AugE"])
    #     data[i, 7] = np.std(info[amp]["num_breakthroughs_AugE"])
    #
    #
    # periods = data[:, 0]
    # period_stds = data[:, 1]
    # num_swallows = data[:, 2]
    # num_swallow_stds = data[:, 3]
    # num_breakthroughs_PreI = data[:, 4]
    # num_breakthroughs_PreI_stds = data[:, 5]
    # num_breakthroughs_AugE = data[:, 6]
    # num_breakthroughs_AugE_stds = data[:, 7]
    #
    # # PLOTTING PERIOD AND STD
    # title = "Swallowing period and standard deviation"
    # xlabel = "Amplitude of the impulse"
    # ylabel = "Period of the swallowing"
    #
    # x = amps[start_idx:]
    # y = periods[start_idx:]
    # error = period_stds[start_idx:]
    # nice_error_bar(x,y,error,title,xlabel,ylabel, title)
    #
    # #PLOTTNG NUMBER OF SWALLOWS
    # title = "Dependence of number of swallows on amplitude"
    # xlabel = "Amplitude of the impulse"
    # ylabel = "Number of swallows"
    # start_idx = 0
    # x = amps[start_idx:]
    # y = num_swallows[start_idx:]
    # error = num_swallow_stds[start_idx:]
    # nice_error_bar(x,y,error,title,xlabel,ylabel, title)
    #
    # # PLOTTNG NUMBER OF Breakthroughs
    # title = "Dependence of number of breathing breakthroughs on amplitude"
    # xlabel = "Amplitude of the impulse"
    # ylabel = "Number of breakthroughs"
    # start_idx = 0
    # x = amps[start_idx:]
    # y = num_breakthroughs_PreI[start_idx:]
    # error = num_breakthroughs_PreI_stds[start_idx:]
    # nice_error_bar(x,y,error,title,xlabel,ylabel, title)
    #
    #
    # # PLOTTNG NUMBER OF Breakthroughs
    # title = "Dependence of number of AugE breakthroughs on amplitude"
    # xlabel = "Amplitude of the impulse"
    # ylabel = "Number of breakthroughs"
    # start_idx = 0
    # x = amps[start_idx:]
    # y = num_breakthroughs_AugE[start_idx:]
    # error = num_breakthroughs_AugE_stds[start_idx:]
    # nice_error_bar(x,y,error,title,xlabel,ylabel, title)

    start_idx = 0
    info = pickle.load(open('../data/info_var_phase.pkl', "rb+"))
    phases = list(info.keys())
    phases.sort()
    data = np.zeros((len(phases), 12))
    for i, phase in (enumerate(phases)):
        data[i, 0] = np.nanmean(info[phase]["Ti0"])
        data[i, 1] = np.std(info[phase]["Ti0"])
        data[i, 2] = np.nanmean(info[phase]["T0"])
        data[i, 3] = np.std(info[phase]["T0"])
        data[i, 4] = np.nanmean(info[phase]["T1"])
        data[i, 5] = np.std(info[phase]["T1"])
        data[i, 6] = np.nanmean(info[phase]["Theta"])
        data[i, 7] = np.std(info[phase]["Theta"])
        data[i, 8] = np.nanmean(info[phase]["Ti1"])
        data[i, 9] = np.std(info[phase]["Ti1"])
        data[i, 10] = np.nanmean(info[phase]["Ti2"])
        data[i, 11] = np.std(info[phase]["Ti2"])

    Phis = np.array(phases)
    Ti_0 = data[:, 0]
    Ti_0_std = data[:, 1]
    T0 = data[:, 2]
    T0_std = data[:, 3]
    T1 = data[:, 4]
    T1_std = data[:, 5]
    Theta = data[:, 6]
    Theta_std = data[:, 7]
    Ti1 = data[:, 8]
    Ti1_std = data[:, 9]
    Ti2 = data[:, 10]
    Ti2_std = data[:, 11]


    # #PLOTTNG Ti_0 from shifts
    # title = "Dependence of $Ti_0$ on shift"
    # xlabel = "Shift, rad"
    # ylabel = "$T_{i0}$"
    # start_idx = 3
    # x = Phis[start_idx:]
    # y = Ti_0[start_idx:]
    # error = Ti_0_std[start_idx:]
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
    #
    # # PLOTTNG Ti_0 from shifts
    # title = "Dependence of $T_0$ on shift"
    # xlabel = "Shift, rad"
    # ylabel = "$T_0$"
    # start_idx = 0
    # x = Phis[start_idx:]
    # y = T0[start_idx:]
    # error = T0_std[start_idx:]
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
    #
    # #PLOTTNG T1 from shifts
    # title = "Dependence of $T_1$ on shift"
    # xlabel = "Shift, rad"
    # ylabel = "$T_{1}$"
    # start_idx = 0
    # x = Phis[start_idx:]
    # y = T1[start_idx:]
    # error = T1_std[start_idx:]
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)

    # # PLOTTNG Phis from shifts
    # title = "Dependence of $\Phi$ on shift"
    # xlabel = "Shift, rad"
    # ylabel = "$\Phi$"
    # start_idx = 0
    # x = Phis[start_idx:]
    # y = T0[start_idx:]
    # error = T0_std[start_idx:]
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)

    # # PLOTTNG Thetas from shifts
    # title = "Dependence of $\Theta$ on shift"
    # xlabel = "Shift, rad"
    # ylabel = "$\Theta$"
    # start_idx = 0
    # x = Phis[start_idx:]
    # y = Theta[start_idx:]
    # error = Theta_std[start_idx:]
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
    #
    # # PLOTTNG Thetas from shifts
    # title = "PRC"
    # xlabel = "Phis (in ms)"
    # ylabel = "$\Delta \Theta"
    # start_idx = 0
    # x = Phis[start_idx:]
    # y = (T1 - T0)/T0[start_idx:]
    # error = np.zeros_like(y)
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
    #
    # title = "Dependence of $(T1s T0s ratio) on \Phi"
    # xlabel = "Phis (in ms)"
    # ylabel = "$\Delta \Theta"
    # start_idx = 0
    # x = Phis[start_idx:]
    # y = (T1/T0)[start_idx:]
    # error = np.zeros_like(y)
    # nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)

    title = "Dependence of \Theta devided by T0"
    xlabel = "Phis (in ms)"
    ylabel = "$\Delta \Theta"
    start_idx = 0
    x = Phis[start_idx:]
    y = (((Theta)/T0) * 2 * np.pi) [start_idx:]
    error = np.zeros_like(y)
    nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)


