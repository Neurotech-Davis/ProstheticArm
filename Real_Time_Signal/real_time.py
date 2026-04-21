import time
import serial
import joblib
import numpy as np
import scipy.signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART' # CHANGE THIS: Your Spresense Mac port
BAUD_RATE = 115200

BOARD_ID = BoardIds.CYTON_BOARD.value 
SERIAL_PORT_BCI = '/dev/cu.usbserial-DP04W035' # CHANGE THIS: Your OpenBCI dongle port

WINDOW_SIZE_SEC = 2.0
COOLDOWN_SEC = 3.0 # Prevents spasming; waits 3 seconds before allowing another trigger
AMPLITUDE_THRESHOLD = 1000.0 # Microvolts threshold for warning

def extract_features(raw_data, sampling_rate, eeg_channels):
    """
    Cleans the raw data, applies artifact rejection, and extracts Alpha power.
    """
    # 1. Isolate just the EEG channels
    eeg_data = raw_data[eeg_channels, :]

    # Subtract the mean of each channel to center the data at 0 uV (Removes DC Offset)
    eeg_centered = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

    # 2. Real-Time Artifact Rejection 
    max_amp = np.max(np.abs(eeg_centered))
    
    if max_amp > AMPLITUDE_THRESHOLD:
        print(f"[{time.strftime('%H:%M:%S')}] NOISE: {max_amp:.2f} uV.")
        
        # HARD LIMIT: If it's over 10,000uV AFTER centering, an electrode is truly disconnected.
        if max_amp > 10000.0:
            print("-> Signal is 'Railed' (disconnected wire). Skipping window to prevent crash.")
            return None 
        else:
            print("-> Forcing data through...")

    # 3. Filtering & Feature Extraction
    features = []
    
    try:
        for ch in range(eeg_data.shape[0]):
            # Force numpy array to be float64 contiguous for C++
            channel_data = np.ascontiguousarray(eeg_centered[ch], dtype=np.float64)
            
            # THE FIX: start_freq = 58.0, stop_freq = 62.0
            DataFilter.perform_bandstop(channel_data, sampling_rate, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0)
            
            # Compute Welch's Power Spectral Density (PSD)
            freqs, psd = scipy.signal.welch(channel_data, fs=sampling_rate, nperseg=sampling_rate)
            
            # Isolate the Alpha band (8-13 Hz)
            alpha_indices = np.logical_and(freqs >= 8.0, freqs <= 13.0)
            alpha_power = np.mean(psd[alpha_indices])
            
            features.append(alpha_power)

        return np.array([features])
        
    except Exception as e:
        print(f"-> Filter failed due to completely corrupted data: {e}. Skipping window.")
        return None

def main():
    print("Loading LDA model and scaler...")
    try:
        lda = joblib.load('lda_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print("Error: Could not find 'lda_model.pkl' or 'scaler.pkl'. Ensure they are in this folder.")
        return

    print(f"Connecting to Spresense on {SERIAL_PORT}...")
    try:
        spresense = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) 
    except Exception as e:
        print(f"Failed to connect to Spresense: {e}")
        return

    print("Connecting to OpenBCI...")
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT_BCI
    board = BoardShim(BOARD_ID, params)
    
    try:
        board.prepare_session()
        board.start_stream()
        print("OpenBCI Stream started! Buffering data...")
        
        sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        samples_per_window = int(WINDOW_SIZE_SEC * sampling_rate)
        
        time.sleep(WINDOW_SIZE_SEC) 
        
        print("\n--- REAL-TIME CLASSIFICATION STARTED ---")
        
        # Variables for debouncing and confidence checks
        last_trigger_time = 0.0
        consecutive_eyes_closed = 0  
        REQUIRED_STREAK = 3          
        CONFIDENCE_THRESHOLD = 0.80  
        
        while True:
            current_time = time.time()
            
            raw_data = board.get_current_board_data(samples_per_window)
            
            if raw_data.shape[1] >= samples_per_window:
                
                features = extract_features(raw_data, sampling_rate, eeg_channels)
                
                if features is None:
                    # Reset the streak if we hit an artifact
                    consecutive_eyes_closed = 0
                    time.sleep(0.1)
                    continue 
                
                scaled_features = scaler.transform(features)
                
                # Get the exact percentage of confidence
                probabilities = lda.predict_proba(scaled_features)[0]
                confidence_class_1 = probabilities[1] # The probability of 'Eyes Closed'
                
                # --- NEW: THE X-RAY HEARTBEAT ---
                # This will print constantly so you know the script isn't frozen
                print(f"Heartbeat | Eyes Closed Confidence: {confidence_class_1*100:.1f}% | Current Streak: {consecutive_eyes_closed}/{REQUIRED_STREAK}")
                
                # Check if the model is confident enough
                if confidence_class_1 >= CONFIDENCE_THRESHOLD:
                    consecutive_eyes_closed += 1
                else:
                    consecutive_eyes_closed = 0 # Reset if they open their eyes
                
                # Trigger Logic - Only fire if streak is met AND cooldown is over
                if consecutive_eyes_closed >= REQUIRED_STREAK:
                    if (current_time - last_trigger_time) > COOLDOWN_SEC:
                        
                        print(f"\n[{time.strftime('%H:%M:%S')}] 🚨 MOTOR TRIGGERED! (Confidence: {confidence_class_1*100:.1f}%) 🚨\n")
                        spresense.write(b'1')
                        
                        last_trigger_time = current_time 
                        consecutive_eyes_closed = 0 # Reset streak after firing
                        
                        time.sleep(0.05)
                        if spresense.in_waiting > 0:
                            print("Spresense:", spresense.readline().decode('utf-8').strip())
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        if 'spresense' in locals() and spresense.is_open:
            spresense.close()
        print("Hardware disconnected gracefully.")

if __name__ == '__main__':
    main()