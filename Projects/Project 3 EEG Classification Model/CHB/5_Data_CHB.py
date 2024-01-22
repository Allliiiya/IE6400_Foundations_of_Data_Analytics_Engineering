import numpy as np
import os
import random
from seizure_info import all_seizure_info

# Paths and parameters
filtered_signals_dir = 'filtered_signals'
output_dir = 'Data_CHB'
fs = 256  # Sampling rate
os.makedirs(output_dir, exist_ok=True)

# Get a list of all filenames
all_files = [f for f in os.listdir(filtered_signals_dir) if f.endswith('.npy')]

# Extract filenames containing seizures from all_seizure_info
seizure_files = set([file_name for file_name in all_seizure_info.keys()])

# Initialize an empty list to collect non-seizure files
non_seizure_files = []

# Iterate through all .npy files
for f in all_files:
    file_base = f.split('.edf')[0]  # Extract the base name of the file
    if file_base not in seizure_files:
        non_seizure_files.append(f)

# Calculate the total number of seizure segments, multiplied by the number of channels
num_seizure_segments = sum(len(info['seizures']) * len([f for f in all_files if f.startswith(file_base.split('.')[0])])
                           for file_base, info in all_seizure_info.items())

# Output the number of seizure segments for verification (the count includes both data and label files)
print(f"Number of seizure files (data and labels): {num_seizure_segments * 2}")

# Create output folders
seizure_output_dir = os.path.join(output_dir, 'seizure')
nonseizure_output_dir = os.path.join(output_dir, 'nonseizure')
os.makedirs(seizure_output_dir, exist_ok=True)
os.makedirs(nonseizure_output_dir, exist_ok=True)

# Save seizure segments and their labels
for file_name, info in all_seizure_info.items():
    seizures = info['seizures']
    for seizure in seizures:
        start, end = seizure[0] * fs, seizure[1] * fs
        for channel_file in all_files:
            if channel_file.startswith(file_name.split('.')[0]):
                signal_data = np.load(os.path.join(filtered_signals_dir, channel_file))
                seizure_segment = signal_data[start:end]
                seizure_label = np.ones(len(seizure_segment))  # Set seizure label to 1
                output_path = os.path.join(seizure_output_dir, f"{channel_file[:-4]}_seizure_{start}_{end}")
                np.save(output_path + '.npy', seizure_segment)
                np.save(output_path + '_label.npy', seizure_label)

# Randomly select an equal number of non-seizure segments
def get_non_seizure_segments(num_segments, segment_length):
    selected_segments = []
    while len(selected_segments) < num_segments:
        random_file = random.choice(non_seizure_files)
        signal_data = np.load(os.path.join(filtered_signals_dir, random_file))
        num_possible_segments = len(signal_data) // segment_length
        if num_possible_segments == 0:
            continue  # Skip if the file is too short

        selected_starts = set()
        while len(selected_segments) < num_segments and len(selected_starts) < num_possible_segments:
            start_idx = random.randint(0, num_possible_segments - 1) * segment_length
            if start_idx in selected_starts:
                continue  # Avoid selecting the same segment again

            end_idx = start_idx + segment_length
            selected_starts.add(start_idx)
            selected_segments.append((random_file, start_idx, end_idx))

    return selected_segments

# Calculate the average length of seizure segments
average_seizure_length = np.mean([end - start for file_info in all_seizure_info.values() for start, end in file_info['seizures']]) * fs

# Get non-seizure segments
non_seizure_segments = get_non_seizure_segments(num_seizure_segments, int(average_seizure_length))

# Save non-seizure segments and their labels
for file_name, start_idx, end_idx in non_seizure_segments:
    signal_data = np.load(os.path.join(filtered_signals_dir, file_name))
    non_seizure_segment = signal_data[start_idx:end_idx]
    non_seizure_label = np.zeros(len(non_seizure_segment))  # Set non-seizure label to 0
    output_path = os.path.join(nonseizure_output_dir, f"{file_name[:-4]}_nonseizure_{start_idx}_{end_idx}")
    np.save(output_path + '.npy', non_seizure_segment)
    np.save(output_path + '_label.npy', non_seizure_label)