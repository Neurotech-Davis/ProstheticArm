import pyxdf

file_path = input("Filename:")

print("Loading XDF file...")
streams, header = pyxdf.load_xdf(file_path)

try:
    marker_strm = [s for s in streams if s['info']['name'][0] == 'PsychoPy_Markers'][0]
    marker_data = marker_strm['time_series']
    
    print("\n--- Unique Markers Found in File ---")
    # Extract the labels and find the unique ones
    unique_markers = set([str(m[0]) for m in marker_data])
    
    if not unique_markers:
        print("WARNING: The PsychoPy marker stream exists, but it is completely empty!")
    else:
        for marker in unique_markers:
            print(f" -> '{marker}'")

except IndexError:
    print("Error: Could not find the 'PsychoPy_Markers' stream at all in this file.")