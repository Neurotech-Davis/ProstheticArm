import pyxdf

# 1. Point this to the exact .xdf file LabRecorder just made
file_path = '/Users/tonysaldana/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'

print(f"Loading {file_path}...\n")

# 2. Load the file
data, header = pyxdf.load_xdf(file_path)

# 3. Loop through the streams to find your markers
for stream in data:
    stream_name = stream['info']['name'][0]
    stream_type = stream['info']['type'][0]
    
    print(f"Found Stream: {stream_name} (Type: {stream_type})")
    
    # If the stream is your PsychoPy markers, print them out!
    if stream_type == 'Markers':
        print("-" * 30)
        print("YOUR EVENT MARKERS:")
        
        timestamps = stream['time_stamps']
        markers = stream['time_series']
        
        for time, marker in zip(timestamps, markers):
            # marker[0] pulls the actual string out of the array
            print(f"Time: {time:.3f} seconds  |  Marker: {marker[0]}")
        
        print("-" * 30 + "\n")
