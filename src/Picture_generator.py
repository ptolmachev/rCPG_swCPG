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

#
info_amp = pickle.load(open("../data/features_var_amp_15_02_2020.pkl",'rb+'))
amps = info_amp['amps']

periods = info_amp['periods']
periods[periods == np.inf] = 10000
periods = np.mean(periods, axis=1)

period_stds = np.mean(info_amp['period_stds'], 1)
period_stds[period_stds == np.inf] = 10000
period_stds = period_stds

rough_periods = np.mean(info_amp['rough_periods'], axis=1)
rough_period_stds = np.std(info_amp['rough_periods'], axis=1)

num_swallows = np.mean(info_amp['num_swallows_s'], axis=1)
num_swallow_stds = np.std(info_amp['num_swallows_s'], axis=1)

num_breakthroughs_AugE = np.mean(info_amp['num_breakthroughs_AugE_s'], axis=1)
num_breakthroughs_AugE_stds = np.std(info_amp['num_breakthroughs_AugE_s'], axis=1)

num_breakthroughs_PreI = np.mean(info_amp['num_breakthroughs_PreI_s'], axis=1)
num_breakthroughs_PreI_stds = np.std(info_amp['num_breakthroughs_PreI_s'], axis=1)

# PLOTTING PERIOD AND STD
title = "Swallowing period and standard deviation"
xlabel = "Amplitude of the impulse"
ylabel = "Period of the swallowing"
start_idx = 0
x = amps[start_idx:]
y = periods[start_idx:]
error = period_stds[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel, title)

# # PLOTTING ROUGH_PERIODS
# title = "Swallowing period (roughly) and standard deviation"
# xlabel = "Amplitude of the impulse"
# ylabel = "Period of the swallowing"
# start_idx = 0
# x = amps[start_idx:]
# y = rough_periods[start_idx:]
# error = rough_period_stds[start_idx:]
# nice_error_bar(x,y,error,title,xlabel,ylabel, title)

#PLOTTNG NUMBER OF SWALLOWS
title = "Dependence of number of swallows on amplitude"
xlabel = "Amplitude of the impulse"
ylabel = "Number of swallows"
start_idx = 0
x = amps[start_idx:]
y = num_swallows[start_idx:]
error = num_swallow_stds[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel, title)

# PLOTTNG NUMBER OF Breakthroughs
title = "Dependence of number of breathing breakthroughs on amplitude"
xlabel = "Amplitude of the impulse"
ylabel = "Number of breakthroughs"
start_idx = 0
x = amps[start_idx:]
y = num_breakthroughs_PreI[start_idx:]
error = num_breakthroughs_PreI_stds[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel, title)


# PLOTTNG NUMBER OF Breakthroughs
title = "Dependence of number of AugE breakthroughs on amplitude"
xlabel = "Amplitude of the impulse"
ylabel = "Number of breakthroughs"
start_idx = 0
x = amps[start_idx:]
y = num_breakthroughs_AugE[start_idx:]
error = num_breakthroughs_AugE_stds[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel, title)

# info_phase = pickle.load(open("../data/features_var_phase_12022020.pkl",'rb+'))
#
# shifts = info_phase['shift']
# Ti_0s = np.roll(info_phase['Ti_0s'], -3, axis=0)
# T0s = np.roll(info_phase['T0s'], -3, axis=0)
# T1s = np.roll(info_phase['T1s'], -3, axis=0)
# Phis = ((info_phase['Phis']/T0s) % 1.0) * 2*np.pi
# Thetas = np.roll(info_phase['Thetas'], -3, axis=0)
# Ti_1s = np.roll(info_phase['Ti_1s'], -3, axis=0)
# Ti_2s = np.roll(info_phase['Ti_2s'], -3, axis=0)

# #PLOTTNG Ti_0 from shifts
# title = "Dependence of $Ti_0$ on shift"
# xlabel = "Shift, rad"
# ylabel = "$T_{i0}$"
# start_idx = 3
# x = np.mean(Phis, axis = 1)[start_idx:]
# y = np.mean(Ti_0s, axis = 1)[start_idx:]
# error = np.std(Ti_0s, axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
#PLOTTNG Ti_0 from shifts
# title = "Dependence of $T_0$ on shift"
# xlabel = "Shift, rad"
# ylabel = "$T_0$"
# start_idx = 0
# x = np.mean(Phis, axis = 1)[start_idx:]
# y = np.mean(T0s, axis = 1)[start_idx:]
# error = np.std(T0s, axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
# #PLOTTNG T1 from shifts
# title = "Dependence of $T_1$ on shift"
# xlabel = "Shift, rad"
# ylabel = "$T_{1}$"
# start_idx = 0
# x = np.mean(Phis, axis = 1)[start_idx:]
# y = np.mean(T1s, axis = 1)[start_idx:]
# error = np.std(T1s, axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
# # PLOTTNG Phis from shifts
# title = "Dependence of $\Phi$ on shift"
# xlabel = "Shift, rad"
# ylabel = "$\Phi$"
# start_idx = 0
# x = np.mean(Phis, axis = 1)[start_idx:]
# y = np.mean(Phis, axis = 1)[start_idx:]
# error = np.std(Phis, axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
# # PLOTTNG Thetas from shifts
# title = "Dependence of $\Theta$ on shift"
# xlabel = "Shift, rad"
# ylabel = "$\Theta$"
# start_idx = 0
# x = np.mean(Phis, axis = 1)[start_idx:]
# y = np.mean(Thetas, axis = 1)[start_idx:]
# error = np.std(Thetas, axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
# # PLOTTNG Thetas from shifts
# title = "Dependence of $\Delta \Theta$ on \Phi"
# xlabel = "Phis (in ms)"
# ylabel = "$\Delta \Theta"
# start_idx = 0
# x = np.roll(np.mean(Phis, axis = 1),-2, axis=0)[start_idx:]
# y = np.mean((T1s - T0s)/T0s, axis = 1)[start_idx:]
# error = np.std((T1s - T0s)/T0s, axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
# title = "Dependence of $(Ti_0s T0s reatio) on \Phi"
# xlabel = "Phis (in ms)"
# ylabel = "$\Delta \Theta"
# start_idx = 0
# x = np.roll(np.mean(Phis, axis = 1), -2, axis=0)[start_idx:]
# y = np.mean((Ti_0s/T0s), axis = 1)[start_idx:]
# error = np.std((Ti_0s/T0s), axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
#
#
# title = "PRC"
# xlabel = "r${\Phi}, rad"
# ylabel = "r$({T_1} - {T_0}) / {T_0}"
# start_idx = 0
# x = np.roll(np.mean(Phis, axis = 1), -2, axis=0)[start_idx:]
# y = np.mean((T1s - T0s) / T0s, axis = 1)[start_idx:]
# error = np.std(((T1s - T0s) / T0s), axis = 1)[start_idx:]
# nice_error_bar_scatter(x,y,error,title,xlabel,ylabel, title)
