import os
import re
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

def main():
    # 1. Setup Tkinter dialog to ask for the folder
    root = tk.Tk()
    root.withdraw() # Hide the main GUI window
    
    # Prompt user to select directory
    folder_path = filedialog.askdirectory(title="Select Folder Containing the Files")
    if not folder_path:
        print("No folder selected. Exiting script.")
        return

    print(f"Scanning folder: {folder_path}\n")

    # 2. Extract stats from the screenshot you provided
    # Ordered with 'report_long' before 'report' so regex doesn't partially match 'report_long' as 'report'
    stats =[
        "gaze_data", 
        "glance_history", 
        "mode_glance_correlation", 
        "mode_intervals", 
        "report_long", 
        "report"
    ]
    
    # Build Regex pattern:
    # Match 1: (stat)
    # Match 2: (filename containing AM.PM)
    # Match 3: (date YYYYMMDD)
    # Match 4: (timestamp HHMMSS)
    # Match 5: (extension csv or json)
    stat_group = "|".join(stats)
    pattern = re.compile(rf"^({stat_group})_(.*AM\.PM.*)_(\d{{8}})_(\d{{6}})\.(csv|json)$", re.IGNORECASE)

    # Dictionary to group pairs together
    # Key: (stat, filename_base, date, extension)
    # Value: list of tuples -> (timestamp_int, original_filename)
    grouped_files = defaultdict(list)

    # 3. Check all filenames in the folder and group them
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            stat = match.group(1)
            file_base = match.group(2)
            date = match.group(3)
            timestamp_str = match.group(4)
            ext = match.group(5)
            
            # Key ignores the timestamp so identical files fall into the same group
            key = (stat, file_base, date, ext)
            grouped_files[key].append((int(timestamp_str), filename))

    # 4. Process groups and rename
    for key, files in grouped_files.items():
        # Ensure we are dealing with pairs 
        if len(files) == 2:
            # Sort the two files by their timestamp integer (Ascending)
            files.sort(key=lambda x: x[0])
            
            lower_ts, lower_ts_filename = files[0]
            higher_ts, higher_ts_filename = files[1]
            
            # Replace 'AM.PM' with 'AM' for the lower timestamp
            new_lower_filename = lower_ts_filename.replace("AM.PM", "AM")
            
            # Replace 'AM.PM' with 'PM' for the higher timestamp
            new_higher_filename = higher_ts_filename.replace("AM.PM", "PM")
            
            # Build absolute paths to execute the rename
            old_path_lower = os.path.join(folder_path, lower_ts_filename)
            new_path_lower = os.path.join(folder_path, new_lower_filename)
            
            old_path_higher = os.path.join(folder_path, higher_ts_filename)
            new_path_higher = os.path.join(folder_path, new_higher_filename)
            
            # Perform OS rename operation
            try:
                os.rename(old_path_lower, new_path_lower)
                print(f"Renamed: {lower_ts_filename}\n      -> {new_lower_filename}")
                
                os.rename(old_path_higher, new_path_higher)
                print(f"Renamed: {higher_ts_filename}\n      -> {new_higher_filename}\n")
            except Exception as e:
                print(f"Error renaming files for group {key}: {e}")
                
        elif len(files) > 2:
            print(f"Warning: Found {len(files)} files for {key[0]} ({key[2]}). Expected exactly 2. Skipping to be safe.")
        else:
            print(f"Warning: Found only 1 file for {key[0]} ({key[2]}). It needs a pair to compare against. Skipping.")

    print("Task completed.")

if __name__ == "__main__":
    main()