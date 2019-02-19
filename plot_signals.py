import matplotlib.pyplot as plt
import pickle

def plot_signals(t, signals, labels, stoptime):

    xlim = [5000,stoptime]
    num_signals = len(signals)
    if num_signals <= 9:
        fig, axs = plt.subplots(num_signals, 1, figsize=(15, 25), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        colors = ['k','r','g','b','y','m','xkcd:tomato','xkcd:lavender', 'xkcd:darkgreen', 'xkcd:plum', 'xkcd:salmon', 'xkcd:coral']
        for i in range(num_signals):
            axs[i].plot(t, signals[i], colors[i],label=labels[i],linewidth = 3)
            axs[i].legend(loc = 1,fontsize = 12)
            axs[i].grid(True)
            ylim = (max(signals[i][xlim[0]:xlim[1]]) if max(signals[i])>0.11 else 1)
            axs[i].axis([xlim[0], xlim[1], 0, 1.1*ylim])
            if i != num_signals-1:
                axs[i].set_xticklabels([])
        plt.show()
    else:
        fig, axs = plt.subplots(9, 1, figsize=(15, 25), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        colors = ['k','r','g','b','y','m','xkcd:tomato','xkcd:lavender', 'xkcd:darkgreen', 'xkcd:plum', 'xkcd:salmon', 'xkcd:coral']
        for i in range(9):
            axs[i].plot(t, signals[i], colors[i],label=labels[i],linewidth = 3)
            axs[i].legend(loc = 1,fontsize = 12)
            axs[i].grid(True)
            ylim = (max(signals[i][xlim[0]:xlim[1]]) if max(signals[i])>0.1 else 1)
            axs[i].axis([xlim[0], xlim[1], 0, 1.1*ylim])
            if i != 9-1:
                axs[i].set_xticklabels([])
        plt.show()

        fig, axs = plt.subplots(num_signals-9, 1, figsize=(15, 25), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        colors = ['k','r','g','b','y','m','xkcd:tomato','xkcd:lavender', 'xkcd:darkgreen', 'xkcd:plum', 'xkcd:salmon', 'xkcd:coral']
        for i in range(num_signals-9):
            axs[i].plot(t, signals[i+9], colors[i],label=labels[i+9],linewidth = 3)
            axs[i].legend(loc = 1,fontsize = 12)
            axs[i].grid(True)
            ylim = (max(signals[i+9][xlim[0]:xlim[1]]) if max(signals[i+9])>0.1 else 1)
            axs[i].axis([xlim[0], xlim[1], 0, 1.1*ylim])
            if i != num_signals-9-1:
                axs[i].set_xticklabels([])
        plt.show()
