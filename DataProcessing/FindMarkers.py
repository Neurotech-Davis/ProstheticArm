# Checks the xdf file to see what markers you have in there.

import pyxdf

# Make sure to put the actual path to your .xdf file here
file_path = input("file name:")
print("Loading XDF file...")
streams, header = pyxdf.load_xdf(file_path)

print(f"Found {len(streams)} streams. Here is what they are called:")
print("-" * 40)

for i, stream in enumerate(streams):
    # Safely extract name and type
    info = stream['info']
    name = info.get('name', ['Unknown'])[0]
    s_type = info.get('type', ['Unknown'])[0]
    channel_count = info.get('channel_count', ['Unknown'])[0]
    
    print(f"Stream {i}:")
    print(f"  Name: '{name}'")
    print(f"  Type: '{s_type}'")
    print(f"  Channels: {channel_count}")
    print("-" * 40)