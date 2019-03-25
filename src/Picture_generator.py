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


info = pickle.load(open("features_var_amp_2.pkl",'rb+'))
amps = info['amps']
periods = info['periods']
period_stds = info['period_stds']
rough_periods = info['rough_periods']
num_swallows = info['num_swallows_s']
num_breakthroughs_AugE = info['num_breakthroughs_AugE_s']
num_breakthroughs_PreI = info['num_breakthroughs_PreI_s']

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