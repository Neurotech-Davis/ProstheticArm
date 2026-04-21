import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# PHASE 1: LOADING AND PREPROCESSING
# ==========================================

# 1. Load the XDF file
file_path = input("What is the file name (e.g., your_file.xdf): ")
print("Loading XDF file...")
streams, header = pyxdf.load_xdf(file_path)

# 2. Extract streams by exact NAME
try:
    eeg_strm = [s for s in streams if s['info']['name'][0] == 'obci_eeg1'][0]
    marker_strm = [s for s in streams if s['info']['name'][0] == 'PsychoPy_Markers'][0]
except IndexError:
    print("Error: Could not find the streams by name. Check your XDF file.")
    exit()

# 3. Convert EEG to MNE Raw format
data = eeg_strm['time_series'].T / 1e6 
sfreq = float(eeg_strm['info']['nominal_srate'][0])

# Name channels Ch1 through Ch8
ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# 4. Apply basic filters
print("Filtering data (1-40Hz Bandpass, 60Hz Notch)...")
raw.filter(1., 40., fir_design='firwin', verbose=False)
raw.notch_filter(60., verbose=False) 

# ==========================================
# PHASE 2: VISUALIZATION AND CLEANING
# ==========================================

# 5. Interactive Time Series (Click bad channels to turn them grey)
print("\nOpening raw trace. Click on any noisy/flatlined channels to mark them as 'bad'.")
# block=True forces Python to wait until the window is closed
raw.plot(duration=10, scalings=dict(eeg=100e-6), title="Raw Trace - Click bad channels!", block=True)

# 6. View Cleaned Combined PSD
print(f"Channels marked as bad: {raw.info['bads']}")
print("Plotting combined PSD (bad channels are automatically excluded)...")
fig = raw.compute_psd(fmax=40).plot(exclude='bads') 
plt.show()

print("Data cleaning visual check complete!")

# ==========================================
# PHASE 3: EPOCHING (SYNCHRONIZING MARKERS)
# ==========================================
print("\n--- Synchronizing Markers and Epoching ---")

# 7. Get the shared LSL timestamps
eeg_times = eeg_strm['time_stamps']
marker_times = marker_strm['time_stamps']
marker_data = marker_strm['time_series']

events = []
# Assign integer codes for ML parsing
event_dict = {'EyesClosed': 1, 'EyesOpen': 2} 

# 8. Loop through every marker PsychoPy sent
for m_time, m_label in zip(marker_times, marker_data):
    label = m_label[0]
    
    # Decide the integer ID
    if 'EyesClosed' in label:
        e_id = event_dict['EyesClosed']
    elif 'EyesOpen' in label:
        e_id = event_dict['EyesOpen']
    else:
        continue # Ignore other markers like 'JawClench' for now
        
    # Find the exact EEG sample index that matches this timestamp
    base_idx = np.searchsorted(eeg_times, m_time)
    
    # 9. Create four sequential 4-second epochs (with a 1s initial buffer)
    for chunk in range(8):
        offset_seconds = 1.0 + (chunk * 2.0) 
        offset_samples = int(offset_seconds * sfreq)
        chunk_start_idx = base_idx + offset_samples
        
        # Ensure the chunk doesn't fall off the end of the recording
        if chunk_start_idx < len(eeg_times):
            events.append([chunk_start_idx, 0, e_id])

# Convert to MNE's required NumPy format
events = np.array(events)
print(f"Created {len(events)} total 2-second epochs.")

# 10. Extract the Epochs
# reject_criteria automatically drops any 4-second chunk exceeding 100uV
reject_criteria = dict(eeg=200e-6) 
epochs = mne.Epochs(raw, events, event_id=event_dict, 
                    tmin=0.0, tmax=2.0, baseline=None, 
                    reject=reject_criteria, preload=True)

print("\n--- Epoch Summary ---")
print(epochs)

# ==========================================
# PHASE 4: FINAL VISUALIZATION & SAVING
# ==========================================

# 11. Visualize the difference in Alpha Power
print("\nPlotting Eyes Closed vs Eyes Open...")

# Grab only Eyes Closed epochs
fig_closed = epochs['EyesClosed'].compute_psd(fmin=2, fmax=30).plot(average=True, spatial_colors=False, show=False)
fig_closed.axes[0].set_title('Eyes Closed')

# Grab only Eyes Open epochs
fig_open = epochs['EyesOpen'].compute_psd(fmin=2, fmax=30).plot(average=True, spatial_colors=False, show=False)
fig_open.axes[0].set_title('Eyes Open')

plt.show()

# 12. Finalize and Save Data
print("\n--- Finalizing and Saving Data ---")

# Permanently delete the channels you clicked/greyed out
if raw.info['bads']:
    print(f"Permanently dropping bad channels: {raw.info['bads']}")
    epochs.drop_channels(raw.info['bads'])
else:
    print("No bad channels were marked.")

# Ask what to name the output file
save_name = input("\nWhat should we name the cleaned save file? (e.g., subject1_clean): ")
file_out = f"{save_name}-epo.fif"

# Save the clean epochs
epochs.save(file_out, overwrite=True)
print(f"Data successfully preprocessed and saved as {file_out}!")