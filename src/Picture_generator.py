import pickle
import numpy as np
from matplotlib import pyplot as plt


def nice_error_bar(x,y,error, title, xlabel,ylabel):
    fig = plt.figure(figsize = (40,20))
    plt.errorbar(x,y, yerr = error , color = 'red',
             ecolor = 'gray', capsize = 3,linestyle = 'dashed', linewidth = 3, alpha = 0.8)

    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.grid(True)
    plt.show()


info_amp = pickle.load(open("features_var_amp_2.pkl",'rb+'))
amps = info_amp['amps']
periods = info_amp['periods']
period_stds = info_amp['period_stds']
rough_periods = info_amp['rough_periods']
num_swallows = info_amp['num_swallows_s']
num_breakthroughs_AugE = info_amp['num_breakthroughs_AugE_s']
num_breakthroughs_PreI = info_amp['num_breakthroughs_PreI_s']

#PLOTTING PERIOD AND STD
# title = "Swallowing period and standard deviation"
# xlabel = "Amplitude of the injected impulse"
# ylabel = "Period of the swallowing"
# start_idx = 67
# x = amps[start_idx:]
# y = np.mean(periods, axis = 1)[start_idx:]
# error = np.mean(period_stds, axis = 1)[start_idx:]
# nice_error_bar(x,y,error,title,xlabel,ylabel)

#PLOTTING ROUGH_PERIODS
# title = "Swallowing period (rougly) and standard deviation"
# xlabel = "Amplitude of the injected impulse"
# ylabel = "Period of the swallowing"
# start_idx = 67
# x = amps[start_idx:]
# y = np.mean(rough_periods, axis = 1)[start_idx:]
# error = np.std(rough_periods, axis = 1)[start_idx:]
# nice_error_bar(x,y,error,title,xlabel,ylabel)

# #PLOTTNG NUMBER OF SWALLOWS
# title = "Dependence of number of swallows on amplitude"
# xlabel = "Amplitude of the injected impulse"
# ylabel = "Number of swallows"
# start_idx = 0
# x = amps[start_idx:]
# y = np.mean(num_swallows, axis = 1)[start_idx:]
# error = np.std(num_swallows, axis = 1)[start_idx:]
# nice_error_bar(x,y,error,title,xlabel,ylabel)

#PLOTTNG NUMBER OF Breakthroughs
# title = "Dependence of number of breathing breakthroughs on amplitude"
# xlabel = "Amplitude of the injected impulse"
# ylabel = "Number of breakthroughs"
# start_idx = 0
# x = amps[start_idx:]
# y = np.mean(num_breakthroughs_PreI, axis = 1)[start_idx:]
# error = np.std(num_breakthroughs_PreI, axis = 1)[start_idx:]
# nice_error_bar(x,y,error,title,xlabel,ylabel)

info_phase = pickle.load(open("features_var_phase_2.pkl",'rb+'))
shifts = info_phase['shift']
Ti_0s = info_phase['Ti_0s']
T0s = info_phase['T0s']
T1s = info_phase['T1s']
Phis = info_phase['Phis']
Thetas = info_phase['Thetas']
Ti_1s = info_phase['Ti_1s']
Ti_2s = info_phase['Ti_2s']

#PLOTTNG Ti_0 from shifts
title = "Dependence of number Ti_0 on shift"
xlabel = "shift"
ylabel = "Ti_0"
start_idx = 0
x = shifts[start_idx:]
y = np.mean(Ti_0s, axis = 1)[start_idx:]
error = np.std(Ti_0s, axis = 1)[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel)

#PLOTTNG Ti_0 from shifts
title = "Dependence of number T0 on shift"
xlabel = "shift"
ylabel = "T0"
start_idx = 0
x = shifts[start_idx:]
y = np.mean(T0s, axis = 1)[start_idx:]
error = np.std(T0s, axis = 1)[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel)

#PLOTTNG T1 from shifts
title = "Dependence of T1 on shift"
xlabel = "shift"
ylabel = "T1"
start_idx = 0
x = shifts[start_idx:]
y = np.mean(T1s, axis = 1)[start_idx:]
error = np.std(T1s, axis = 1)[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel)

# PLOTTNG Phis from shifts
title = "Dependence of Phis on shift"
xlabel = "shift"
ylabel = "Phis"
start_idx = 0
x = shifts[start_idx:]
y = np.mean(Phis, axis = 1)[start_idx:]
error = np.std(Phis, axis = 1)[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel)

# PLOTTNG Thetas from shifts
title = "Dependence of Thetas on shift"
xlabel = "shift"
ylabel = "Thetas"
start_idx = 0
x = shifts[start_idx:]
y = np.mean(Thetas, axis = 1)[start_idx:]
error = np.std(Thetas, axis = 1)[start_idx:]
nice_error_bar(x,y,error,title,xlabel,ylabel)