import scipy.io as sio
from matplotlib import pyplot as plt
mat_contents = sio.loadmat('/home/pavel/Documents/0Research/Breathing system/Swallowing/Experimental recordings/OneDrive_1_3-15-2019/20170609_2_f_p23_sln_t2.mat')

# print(type(mat_contents))
# print(mat_contents.keys())
# print(mat_contents['data'].shape)
# print(mat_contents['datastart'])
# print(mat_contents['dataend'])
# print(mat_contents['titles'])
# print(mat_contents['data'])

series = mat_contents['data'][0][int(mat_contents['datastart'][3]) : int(mat_contents['dataend'][3])]

print(series)
fig = plt.figure(figsize = (10,5))
plt.plot(series, 'k-', linewidth = 2, alpha = 0.7)
plt.grid(True)
plt.show()