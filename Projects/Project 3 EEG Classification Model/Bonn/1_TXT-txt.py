import os

# Set the folder path
folder_path = 'C:/Users/surpriseX/Desktop/HW/6400/project3/BonnEEG/N'

# List all files in the folder
files = os.listdir(folder_path)

# Loop through each file
for file_name in files:
    # Check if the file extension is uppercase .TXT
    if file_name.endswith('.TXT'):
        # Construct the old and new file paths
        old_file = os.path.join(folder_path, file_name)
        new_file = os.path.join(folder_path, file_name[:-4] + '.txt')

        # Rename the file
        os.rename(old_file, new_file)

