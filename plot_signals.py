import matplotlib.pyplot as plt
import pickle

# fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .5, wspace=.001)
#
# axs = axs.ravel()
#
# for i in range(10):
#
#     axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
#     axs[i].set_title(str(250+i))

# t,s1,s2,s3,s4 = pickle.load(open("./reference_signals.pkl","rb+"))
# signals = [s1,s2,s3,s4]

def plot_signals(t, signals):

    num_signals = len(signals)
    fig, axs = plt.subplots(num_signals, 1, figsize=(15, 25), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    colors = ['k','r','g','b','y','m','c','xkcd:olive', 'xkcd:grey', 'xkcd:plum', 'xkcd:salmon', 'xkcd:coral']
    for i in range(num_signals):
        axs[i].plot(t, signals[i], colors[i],label=r'$f(v_1)$',linewidth = 3)
        axs[i].legend(loc = 1,fontsize = 12)
        axs[i].grid(True)
        axs[i].axis([5000, 20000, 0, 1])
        if i != num_signals-1:
            axs[i].set_xticklabels([])

    plt.show()

# plot_signals(t,signals)