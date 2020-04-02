import pickle
from matplotlib import pyplot as plt
from utils.gen_utils import get_project_root
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from tqdm.auto import tqdm
import numpy as np

# need to load original data
# apply hilbert transform
# get the phase curve of the recording
### PLOTTING CHUNKS
data_path = str(get_project_root()) + "/data"
img_path = str(get_project_root()) + "/img"
data_folders = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc', '2019-09-05_12-26-14_prc']
for i in (range(len(data_folders))):
    file_load = f'{data_path}/sln_prc_chunked/{data_folders[i]}/chunked.pkl'
    data = pickle.load(open(file_load, 'rb+'))
    list_chunks = list(data.keys())
    for chunk_num in tqdm(list_chunks):
        data_chunk = data[chunk_num]
        analytic_signal_PNA = hilbert(savgol_filter(data_chunk['PNA'], 71, 3))
        amplitude_envelope = np.abs(analytic_signal_PNA)
        offset = 0
        instantaneous_phase = np.unwrap(np.arctan( (np.imag(analytic_signal_PNA) - offset)/(np.real(analytic_signal_PNA) - offset))) #np.angle(analytic_signal_PNA)

        # amplitude_envelope = np.abs(analytic_signal_PNA)
        # amplitude_envelope = (amplitude_envelope - np.min(amplitude_envelope)) / (
        #             np.max(amplitude_envelope) - np.min(amplitude_envelope)) * 2 - 1
        # offset = 0
        # instantaneous_phase = np.arctan(
        #     (np.imag(analytic_signal_PNA) - offset) / (np.real(analytic_signal_PNA) - offset))

        # from scipy.signal import periodogram
        # from scipy.signal import butter, filtfilt

        # plt.plot(periodogram(data_chunk["PNA"])[1])
        # PNA_filtered = butter_lowpass_filter(data_chunk["PNA"], 200, 30000)
        # # plt.plot(PNA_filtered)
        # analytic_signal_PNA = hilbert(PNA_filtered)
        # amplitude_envelope = np.abs(analytic_signal_PNA)
        # instantaneous_phase = np.unwrap(np.angle(analytic_signal_PNA))
        # plt.plot(amplitude_envelope)
        # plt.plot(PNA_filtered)

        # plt.plot(data_chunk["PNA"] + 1)
        # plt.plot(instantaneous_phase)  # % (np.pi * 2)
        # plt.plot(filtered_signal)
        # plt.plot(data_chunk["PNA"])
        # plt.plot(savgol_filter(data_chunk["PNA"], 71, 3))

        x = 1