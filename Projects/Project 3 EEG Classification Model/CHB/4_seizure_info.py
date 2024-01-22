import re
import os


def parse_summary_file(summary_file_path):
    seizure_info = {}
    with open(summary_file_path, 'r') as file:
        content = file.read()

        file_pattern = re.compile(
            r'File Name: (.+?)\s+.*?Number of Seizures in File: (\d+)', re.DOTALL)
        file_matches = list(file_pattern.finditer(content))

        for i, match in enumerate(file_matches):
            file_name = match.group(1).strip()
            num_seizures = int(match.group(2).strip())
            if num_seizures > 0:  # Only include files where seizures occur
                if i + 1 < len(file_matches):
                    file_block = content[match.start():file_matches[i + 1].start()]
                else:
                    file_block = content[match.start():]

                # Updated pattern to match seizures without a number
                seizure_pattern = re.compile(
                    r'Seizure\s*(\d+)?\s*Start Time: (\d+) seconds\s+.*?Seizure\s*(\d+)?\s*End Time: (\d+) seconds',
                    re.DOTALL)
                seizures = [(int(start), int(end)) for _, start, _, end in seizure_pattern.findall(file_block)]
                seizure_info[file_name] = {
                    'num_seizures': num_seizures,
                    'seizures': seizures
                }

    return seizure_info


summary_files_dir = 'C:\\Users\\surpriseX\\Desktop\\HW\\6400\\project3\\CHB-MIT\\summary_files_directory'
all_seizure_info = {}

for i in range(1, 25):
    summary_file_name = f"chb{i:02}-summary.txt"
    summary_file_path = os.path.join(summary_files_dir, summary_file_name)
    if os.path.exists(summary_file_path):  # Check if the file exists before trying to parse
        seizure_info = parse_summary_file(summary_file_path)

        # Only update the all_seizure_info dictionary if seizures were found in the file
        if seizure_info:
            all_seizure_info.update(seizure_info)

# Print the dictionary containing seizure information for verification
# for file, info in all_seizure_info.items():
    # print(f"{file}: {info}")
