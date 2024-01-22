import numpy as np
import matplotlib.pyplot as plt

EEG_data = np.loadtxt('C:/Users/surpriseX/Desktop/HW/6400/project3/preBonn/F/F021.txt')

# Create timeline
time_axis = np.arange(0, len(EEG_data)) / 173.61

# Mapping brain waves
plt.figure(figsize=(10, 4))
plt.plot(time_axis, -EEG_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Signal')
plt.grid(True)
plt.show()
