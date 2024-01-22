import numpy as np
from scipy.signal import butter, lfilter
import os

# Set parameters for low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Set folder and file parameters
base_path = 'C:/Users/surpriseX/Desktop/HW/6400/project3/BonnEEG/'
folders = ['F', 'N', 'O', 'S', 'Z']
output_folder = 'C:/Users/surpriseX/Desktop/HW/6400/project3/preBonn/'
# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start preprocessing
for folder in folders:
    # Create the output directory if it doesn't exist
    folder_output_path = os.path.join(output_folder, folder)
    if not os.path.exists(folder_output_path):
        os.makedirs(folder_output_path)

    # Iterate through files in each folder
    folder_path = os.path.join(base_path, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.TXT'):
            file_path = os.path.join(folder_path, filename)
            # Read the data
            data = np.loadtxt(file_path)

            # Apply low-pass filter
            filtered_data = lowpass_filter(data, cutoff=40, fs=173.61)

            # Save the preprocessed data to a new file
            output_file_path = os.path.join(folder_output_path, filename)
            np.savetxt(output_file_path, filtered_data)

# Print output to confirm completion
print("Preprocessing complete. Filtered data saved to: " + output_folder)
