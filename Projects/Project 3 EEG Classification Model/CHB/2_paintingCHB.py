import mne
import matplotlib.pyplot as plt

file_path = 'C:/Users/surpriseX/Desktop/HW/6400/project3/CHB-MIT/chb01/chb01_01.edf'

# Load EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

# Select a portion of the data to plot
start, stop = raw.time_as_index([100, 200]) # From 100 seconds to 200 seconds
data, times = raw[:, start:stop]

# Plot EEG waveforms
plt.figure(figsize=(10, 8))
plt.plot(times, data.T)
plt.xlabel('Time (s)')
plt.ylabel('EEG data (uV)')
plt.title('EEG Waveforms')
plt.show()
