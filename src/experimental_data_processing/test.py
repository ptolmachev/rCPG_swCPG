import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show
import numpy as np
ion()

x = np.linspace(1,10,100)
y = np.sin(x)
pos = []
def onclick(event):
    x = event.xdata
    y = event.ydata
    pos.append(event.xdata)
    plt.plot(x,y, 'ro')

fig = plt.figure()
plot(x,y)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
show(block=True)
fig.canvas.mpl_disconnect(cid)
print(pos)
