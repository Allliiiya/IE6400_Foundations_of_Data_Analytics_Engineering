import os
import pyedflib

# Define the root directory of the dataset
data_dir = r'C:\Users\surpriseX\Desktop\HW\6400\project3\CHB-MIT'

# List all subfolders, i.e., chb01 to chb24
patient_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Iterate through each patient's folder and print out information about EDF files and summary files
for patient_folder in patient_folders:
    folder_path = os.path.join(data_dir, patient_folder)
    edf_files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]
    summary_file = [f for f in os.listdir(folder_path) if f.endswith('-summary.txt')]

    print(f"Patient Folder: {patient_folder}")
    print(f"EDF Files: {edf_files}")
    print(f"Summary File: {summary_file}")
    print("="*50)  # Print a separator to distinguish different patient folders

# Here is an example using the first EDF file of the first patient
sample_edf_file = os.path.join(data_dir, patient_folders[0], 'chb01_01.edf')
f = pyedflib.EdfReader(sample_edf_file)

# Print out some information about the EDF file
print(f"File name: {sample_edf_file}")
print(f"Number of channels: {f.signals_in_file}")
print(f"Signal labels: {f.getSignalLabels()}")
print(f"Data records in file: {f.datarecords_in_file}")
print(f"Duration of a data record: {f.getFileDuration()}")
print(f"Sampling frequency of each signal: {f.getSampleFrequencies()}")

# Close the file
f._close()