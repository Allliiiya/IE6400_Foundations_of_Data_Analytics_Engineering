import os
import pyedflib
from scipy.signal import butter, lfilter
import numpy as np
import gc

# Define the filter function
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

# Set the folder path and filter parameters
data_dir = 'C:\\Users\\surpriseX\\Desktop\\HW\\6400\\project3\\CHB-MIT'
lowcut = 0.5
highcut = 49.0
fs = 256  # Sampling frequency

# Directory for storing filtered signals
filtered_signals_dir = 'filtered_signals'
os.makedirs(filtered_signals_dir, exist_ok=True)

# Iterate through each subfolder in the data directory
for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.edf'):
            edf_path = os.path.join(subdir, file)
            try:
                with pyedflib.EdfReader(edf_path) as f:
                    n_channels = f.signals_in_file
                    signal_labels = f.getSignalLabels()
                    for i in range(n_channels):
                        signal = f.readSignal(i).astype('float32')  # Convert the signal to float32
                        filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=4)
                        # Save the filtered signal to a file
                        np.save(os.path.join(filtered_signals_dir, f"{file}_{signal_labels[i]}.npy"), filtered_signal)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
            finally:
                # Force garbage collection
                gc.collect()