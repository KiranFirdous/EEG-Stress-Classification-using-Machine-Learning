# Importing necessary libraries
import mne
import numpy as np
import scipy as sp
import pandas as pd
from glob import glob
from IPython.display import display
import matplotlib.pyplot as plt
import math
from skimage.restoration import denoise_wavelet
from scipy.signal import savgol_filter
from scipy.signal import medfilt
import seaborn as sns
import pywt

# Load and display the first subject's data
d_frame = pd.read_csv("Data_30_EC/subject01_0.csv")
d_frame

# Visualize the first three seconds of EEG data
plt.plot(d_frame['Raw Data'])
plt.xlabel('Time')
plt.title('The first three seconds of subject_01 FP1 Channel')
d_frame.head()

# Plot the distribution of EEG values
d_frame['Raw Data'].hist()
plt.xlabel('Time')
plt.title('The Distribution of the EEG FP1')

# Get all data file paths
Raw_data_paths = sorted(glob("Data_30_EC/*"))
len(Raw_data_paths)
Raw_data_paths

def create_student_data_dict(Raw_data_paths):
    """
    This function creates a dictionary that contains students as keys 
    and the EEG data as values.
    INPUT -----> the sorted list of student' EEG files.
    OUTPUT ----> A python Dict that contains 
    keys: Students 
    Values: Data
    """
    raw_dic = {}
    for path_index in range(1, 31):
        key = Raw_data_paths[path_index][-15:-4]  # extract subjectxx-x from filenames
        data_frame = pd.read_csv(Raw_data_paths[path_index])
        raw_dic[key] = data_frame
    return raw_dic

raw_dic = create_student_data_dict(Raw_data_paths)

# Check structure of the dictionary
print('raw_dic contains %d DataFrame' % len(raw_dic))
raw_dic['subject01_0']

number_of_channels = raw_dic['subject01_0'].shape[1]
number_of_channels

names_of_channels = raw_dic['subject02_1'].columns
names_of_channels

# Signal Processing Functions
# Band pass filter between 0.5 and 40 Hz
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Median filter
from scipy.signal import medfilt

def median(signal):
    """Apply median filter to signal"""
    array = np.array(signal)
    med_filtered = sp.signal.medfilt(array, kernel_size=3)
    return med_filtered

# Notch filter applied at 50Hz
def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    fs = 1/time
    nyq = fs/2.0
    low = freq - band/2.0
    high = freq + band/2.0
    low = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def SignalProcessing(raw_dic):
    """
    This function implements the signal processing pipeline through:
    1- Median Filter 
    2- Band pass filter 
    3- Wavelet denoise
    4- Savitzky-Golay filter
    
    INPUT -------> the raw signals
    OUTPUT ------> A dictionary that contains the denoised signals 
    """
    time_sig_dic = {}  # Dictionary for time domain signals
    raw_dic_keys = sorted(raw_dic.keys())
    
    for key in raw_dic_keys:
        raw_df = raw_dic[key]
        time_sig_df = pd.DataFrame()
        
        for column in raw_df.columns:
            t_signal = np.array(raw_df[column])
            med_filtred = median(t_signal)
            
            fs = 50
            lowcut = 0.05
            highcut = 5
            band_pass = butter_bandpass_filter(med_filtred, lowcut, highcut, fs, order=5)
            
            wavelet_denoise = denoise_wavelet(band_pass, method='BayesShrink', mode='hard', 
                                              wavelet='sym9', wavelet_levels=5, rescale_sigma=True)
            
            clean_signals = savgol_filter(wavelet_denoise, 1111, 3, mode='wrap')
            time_sig_df[column] = clean_signals
        
        time_sig_dic[key] = time_sig_df
    
    return time_sig_dic

time_sig_dic = SignalProcessing(raw_dic)

# Calculate signal lengths
time_list = []
for i in range(0, 30):
    time = (time_sig_dic[Raw_data_paths[i][-15:-4]].index.values[-1])
    time_list.append(time)

data = np.array(time_list)
time_length = pd.DataFrame(data=data, columns=['Signl length'])
time_length

# Helper functions for string normalization
def normalize5(number):
    """Add leading zeros to make string length 5"""
    stre = str(number)
    if len(stre) < 5:
        l = len(stre)
        for i in range(0, 5 - l):
            stre = "0" + stre
    return stre

def normalize2(number):
    """Add leading zeros to make string length 2"""
    stre = str(number)
    if len(stre) < 2:
        stre = "0" + stre
    return stre

def Windowing(time_sig_dic):
    """
    This Function is used to segment the data into small windows.
    Window size: 4 seconds (200 samples at 50Hz)
    
    INPUT ----> The denoised signal dictionary 
    OUTPUT ---> A dictionary that contains the windows 
    """
    window_dict = {}
    columns = time_sig_dic['subject02_1'].columns
    
    # Subject IDs and their states (0 or 1)
    subject_ids = list(range(1, 31))
    states = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    
    for subject_id, state in zip(subject_ids, states):
        file_key = 'subject' + normalize2(int(subject_id)) + '_' + str(state)
        dic_update = raw_dic[file_key]
        
        for sig_time in range(0, len(time_length)):
            sig_time_length = (time_length['Signl length'][sig_time])
            window_ID = 0
            
            for cursor in range(0, (sig_time_length - 199), 50):
                end_point = cursor + 200
                data = np.array(dic_update.iloc[cursor:end_point])
                window = pd.DataFrame(data=data, columns=columns)
                key = 't_W' + normalize5(window_ID) + '_' + file_key
                window_dict[key] = window
                window_ID = window_ID + 1
    
    return window_dict

window_dict = Windowing(time_sig_dic)

# Clean up empty frames
new_frames = {k: v for (k, v) in window_dict.items() if not v.empty}
len(new_frames)
sorted(new_frames.keys())[0]

# Fourier Transform Functions
from scipy import fftpack
from numpy.fft import *

def fast_fourier_transform_one_signal(t_signal):
    """Apply FFT to a single signal"""
    complex_f_signal = fftpack.fft(t_signal)
    amplitude_f_signal = np.abs(complex_f_signal)
    return amplitude_f_signal

def fast_fourier_transform(t_window):
    """Apply FFT to all columns in a window"""
    f_window = pd.DataFrame()
    for column in t_window.columns:
        t_signal = np.array(t_window[column])
        f_signal = np.apply_along_axis(fast_fourier_transform_one_signal, 0, t_signal)
        f_window["freq_" + column[0:]] = f_signal
    return f_window

# Apply FFT to all windows
f_window_dict = {'f' + key[1:]: t_w1_df.pipe(fast_fourier_transform) for key, t_w1_df in new_frames.items()}
fnew_frames = {k: v for (k, v) in f_window_dict.items() if not v.empty}
len(fnew_frames)

# Time-Frequency Domain using Wavelet Transform
def tf(t_freq_signal):
    """Apply Discrete Wavelet Transform"""
    (cA, cD) = pywt.dwt(t_freq_signal, 'db1')
    x = np.concatenate((cD, cA), axis=0)
    return x

time_freq_dic = {}
time_dic_keys = sorted(new_frames.keys())

for k in time_dic_keys:
    time_df = new_frames[k]
    time_freq_df = pd.DataFrame()
    
    for c in time_df.columns:
        t_freq_signal = np.array(time_df[c])
        sum_of_coff = np.apply_along_axis(tf, 0, t_freq_signal)
        time_freq_df['time_freq' + c] = sum_of_coff
    
    time_freq_dic[k] = time_freq_df

t_f_newframes = {k: v for (k, v) in time_freq_dic.items() if not v.empty}
t_f_newframes['t_W00000_subject01_0'].columns[0]

# Feature Extraction Functions
def mean_axial(df):
    array = np.array(df)
    mean_vector = array.mean(axis=0)
    return mean_vector

def std_axial(df):
    array = np.array(df)
    std_vector = array.std(axis=0)
    return std_vector

from statsmodels.robust import mad as median_deviation

def mad_axial(df):
    array = np.array(df)
    mad_vector = median_deviation(array, axis=0)
    return mad_vector

def max_axial(df):
    array = np.array(df)
    max_vector = array.max(axis=0)
    return max_vector

def min_axial(df):
    array = np.array(df)
    min_vector = array.min(axis=0)
    return min_vector

from scipy.stats import iqr as IQR

def IQR_axial(df):
    array = np.array(df)
    IQR_vector = np.apply_along_axis(IQR, 0, array)
    return IQR_vector

from scipy.stats import entropy

def entropy_axial(df):
    array = np.array(df)
    entropy_vector = np.apply_along_axis(entropy, 0, abs(array))
    return entropy_vector

def t_energy_axial(df):
    array = np.array(df)
    energy_vector = (array**2).sum(axis=0)
    return energy_vector

# Time Features Pipeline
def t_axial_features_generation(t_window):
    axial_columns = t_window.columns[0]
    t_axial_features = []
    
    mean_vector = list(mean_axial(t_window))
    std_vector = list(std_axial(t_window))
    mad_vector = list(mad_axial(t_window))
    max_vector = list(max_axial(t_window))
    min_vector = list(min_axial(t_window))
    energy_vector = list(t_energy_axial(t_window))
    IQR_vector = list(IQR_axial(t_window))
    entropy_vector = list(entropy_axial(t_window))
    
    t_3axial_vector = mean_vector + std_vector + mad_vector + max_vector + min_vector + energy_vector + IQR_vector + entropy_vector
    t_axial_features = t_axial_features + t_3axial_vector
    
    return t_axial_features

def time_features_names():
    """Generate time domain feature names"""
    t_axis_signals = [['EEG ']]
    t_one_input_features_name1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    t_one_input_features_name2 = ['_energy()', '_iqr()', '_entropy()']
    
    features = []
    for columns in t_axis_signals:
        for feature in t_one_input_features_name1:
            for column in columns:
                newcolumn = column + feature
                features.append(newcolumn)
        for feature in t_one_input_features_name2:
            for column in columns:
                newcolumn = column + feature
                features.append(newcolumn)
    
    return features

# Frequency Features
def f_energy_axial(df):
    array = np.array(df)
    energy_vector = (array**2).sum(axis=0) / float(len(array))
    return energy_vector

from scipy.stats import kurtosis
from scipy.stats import skew

def f_skewness_and_kurtosis_axial(df):
    array = np.array(df)
    skew_X = skew(array)
    kur_X = kurtosis(array)
    skew_kur_3axial_vector = [skew_X, kur_X]
    return skew_kur_3axial_vector

def t_f_skewness_and_kurtosis_axial(df):
    array = np.array(df)
    skew_X = skew(array)
    kur_X = kurtosis(array)
    skew_kur_3axial_vector = [skew_X, kur_X]
    return skew_kur_3axial_vector

def f_axial_features_generation(f_window):
    axial_columns = f_window.columns[0]
    f_all_axial_features = []
    
    mean_vector = list(mean_axial(f_window))
    std_vector = list(std_axial(f_window))
    mad_vector = list(mad_axial(f_window))
    max_vector = list(max_axial(f_window))
    min_vector = list(min_axial(f_window))
    IQR_vector = list(IQR_axial(f_window))
    entropy_vector = list(entropy_axial(f_window))
    energy_vector = list(f_energy_axial(f_window))
    skewness_and_kurtosis_vector = f_skewness_and_kurtosis_axial(f_window)
    
    f_3axial_features = mean_vector + std_vector + mad_vector + max_vector + min_vector + energy_vector + IQR_vector + entropy_vector + skewness_and_kurtosis_vector
    f_all_axial_features = f_all_axial_features + f_3axial_features
    
    return f_all_axial_features

def t_f_axial_features_generation(t_f_window):
    axial_columns = t_f_window.columns[0]
    t_f_all_axial_features = []
    
    mean_vector = list(mean_axial(t_f_window))
    std_vector = list(std_axial(t_f_window))
    mad_vector = list(mad_axial(t_f_window))
    max_vector = list(max_axial(t_f_window))
    min_vector = list(min_axial(t_f_window))
    IQR_vector = list(IQR_axial(t_f_window))
    entropy_vector = list(entropy_axial(t_f_window))
    skewness_and_kurtosis_vector = t_f_skewness_and_kurtosis_axial(t_f_window)
    
    f_3axial_features = mean_vector + std_vector + mad_vector + max_vector + min_vector + IQR_vector + entropy_vector + skewness_and_kurtosis_vector
    t_f_all_axial_features = t_f_all_axial_features + f_3axial_features
    
    return t_f_all_axial_features

def frequency_features_names():
    """Generate frequency domain feature names"""
    axial_signals = [['EEG ']]
    f_one_input_features_name1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    f_one_input_features_name2 = ['_energy()', '_iqr()', '_entropy()']
    f_one_input_features_name3 = ['_skewness()', '_kurtosis()']
    
    frequency_features_names = []
    for columns in axial_signals:
        for feature in f_one_input_features_name1:
            for column in columns:
                newcolumn = column + feature
                frequency_features_names.append(newcolumn)
        for feature in f_one_input_features_name2:
            for column in columns:
                newcolumn = column + feature
                frequency_features_names.append(newcolumn)
        for column in columns:
            for feature in f_one_input_features_name3:
                newcolumn = column + feature
                frequency_features_names.append(newcolumn)
    
    return frequency_features_names

def t_frequency_features_names():
    """Generate time-frequency domain feature names"""
    axial_signals = [['EEG ']]
    f_one_input_features_name1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    f_one_input_features_name2 = ['_iqr()', '_entropy()']
    f_one_input_features_name3 = ['_skewness()', '_kurtosis()']
    
    time_frequency_features_names = []
    for columns in axial_signals:
        for feature in f_one_input_features_name1:
            for column in columns:
                newcolumn = column + feature
                time_frequency_features_names.append(newcolumn)
        for feature in f_one_input_features_name2:
            for column in columns:
                newcolumn = column + feature
                time_frequency_features_names.append(newcolumn)
        for column in columns:
            for feature in f_one_input_features_name3:
                newcolumn = column + feature
                time_frequency_features_names.append(newcolumn)
    
    return time_frequency_features_names

# Concatenate all feature names
all_columns = time_features_names() + frequency_features_names() + t_frequency_features_names() + ['state', 'subject']

def Dataset_Generation_PipeLine(t_dic, f_dic, t_f_dic):
    """Generate final dataset from time, frequency, and time-frequency features"""
    final_Dataset = pd.DataFrame(data=[], columns=all_columns)
    
    for i in range(len(t_dic)):
        t_key = sorted(t_dic.keys())[i]
        f_key = sorted(f_dic.keys())[i]
        t_f_key = sorted(t_f_dic.keys())[i]
        
        t_window = t_dic[t_key]
        f_window = f_dic[f_key]
        t_f_window = t_f_dic[t_f_key]
        
        window_user_id = int(t_key[-4:-2])
        window_activity_id = int(t_key[-1])
        
        time_features = t_axial_features_generation(t_window)
        frequency_features = f_axial_features_generation(f_window)
        time_freq_features = t_f_axial_features_generation(t_f_window)
        
        row = time_features + frequency_features + time_freq_features + [int(window_activity_id), int(window_user_id)]
        free_index = len(final_Dataset)
        final_Dataset.loc[free_index] = row
    
    return final_Dataset

# Generate the final dataset
Dataset = Dataset_Generation_PipeLine(new_frames, fnew_frames, t_f_newframes)

# Display dataset information
print('The shape of Dataset is :', Dataset.shape)
display(Dataset.describe())
display(Dataset.head(10))

# Save the dataset to CSV
path = "./EC_EEG_Clean_Data_30.csv"
Dataset.to_csv(
    path_or_buf=path,
    na_rep='NaN',
    columns=None,
    header=True,
    index=False,
    mode='w',
    encoding='utf-8',
    line_terminator='\n'
)
