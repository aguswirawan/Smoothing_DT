import os
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import glob

def read_file(file):
    data = sio.loadmat(file)
    return data

def GaussianSmoothing(base_signal, sigma=1.0):
    return gaussian_filter1d(base_signal, sigma=sigma)

def Gaussian(base_signal, smoothing_sigma=1.0):
    base_signal = GaussianSmoothing(base_signal, sigma=smoothing_sigma)
    data_mean = np.mean(base_signal)
    data_std = np.std(base_signal)
    data_norm = norm.pdf(base_signal, data_mean,data_std)

    return np.array(data_norm)

def decompose(data):
    frequency = 128
    smoothed_signals = np.empty([0, 384])

    for trial in range(24):
        for channel in range(14):
            base_signal = data[:, trial][0][:384, channel]
            smoothed_signal = Gaussian(base_signal, smoothing_sigma=1.0)
            smoothed_signals = np.vstack([smoothed_signals, smoothed_signal.reshape(1, -1)])

    print("smoothed_signals shape:", smoothed_signals.shape)
    return smoothed_signals

def get_labels(data_labels, data_trial):
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    final_dominance_labels = np.empty([0])

    for trial in range(24):
        data_label = data_labels[:, trial][0][0].copy()
        data_trial_row = (data_trial[:, trial][0].shape[0] // 128) + 1 - 5

        valence_labels = np.array([data_label[1] == 1] * data_trial_row)
        arousal_labels = np.array([data_label[0] == 1] * data_trial_row)
        dominance_labels = np.array([data_label[2] == 1] * data_trial_row)

        final_valence_labels = np.append(final_valence_labels, valence_labels)
        final_arousal_labels = np.append(final_arousal_labels, arousal_labels)
        final_dominance_labels = np.append(final_dominance_labels, dominance_labels)

    print("labels_valence:", final_valence_labels.shape)
    print("labels_arousal:", final_arousal_labels.shape)
    print("labels_dominance:", final_dominance_labels.shape)
    return final_valence_labels, final_arousal_labels, final_dominance_labels

if __name__ == '__main__':
    dataset_dir = "All Movement Dataset/"
    result_dir = "1D_Smoothing/"
    if os.path.isdir(result_dir) == False:
        os.makedirs(result_dir)
    for file in sorted(glob.glob(dataset_dir + '*.mat')):
        filename = file.split('\\')[-1]
        print("processing: ", filename, "......")
        file_path = os.path.join(dataset_dir, filename)
        data = read_file(file)
        smoothed_signals = decompose(data['joined_data1'].copy())
        valence_labels, arousal_labels, dominance_labels = get_labels(data['labels_selfassessment'].copy(), data['joined_data1'].copy())
        sio.savemat(result_dir + "DE_" + filename, {
            "smoothed_signals": smoothed_signals,
            "valence_labels": valence_labels,
            "arousal_labels": arousal_labels,
            "dominance_labels": dominance_labels
        })
