import mne
import numpy as np
import joblib 
import scipy.signal 
from brainflow.data_filter import DataFilter, FilterTypes 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# PHASE 1: LOAD AND COMBINE DATA
# ==========================================
print("--- BCI Classifier: Loading Data ---")
epochs_list = []

# 1. Loop to allow you to input multiple .fif files
while True:
    file_path = input("Enter a .fif file name (or type 'done' to continue): ")
    if file_path.lower() == 'done':
        break
    try:
        epo = mne.read_epochs(file_path, preload=True, verbose=False)
        epochs_list.append(epo)
        print(f"Successfully loaded {file_path}. Total files queued: {len(epochs_list)}")
    except FileNotFoundError:
        print("File not found. Please check the name and try again.")

if len(epochs_list) == 0:
    print("No data loaded. Exiting...")
    exit()

# 2. Combine all files into one massive dataset
all_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
print(f"\nTotal epochs loaded across all files: {len(all_epochs)}")

# ==========================================
# PHASE 2: FEATURE EXTRACTION (Synchronized & Scaled!)
# ==========================================
print("\n--- Extracting Alpha Power Features (Synchronized with Real-Time) ---")

# --- THE FIX: Multiply by 1 million to convert MNE's Volts into BrainFlow's Microvolts ---
data = all_epochs.get_data() * 1e6 
sampling_rate = int(all_epochs.info['sfreq']) # Get sampling rate from file

features_list = []

# Loop through every single epoch
for i in range(data.shape[0]):
    epoch_data = data[i] 

    # 1. REMOVE DC OFFSET (Exactly like the real-time script)
    eeg_centered = epoch_data - np.mean(epoch_data, axis=1, keepdims=True)

    epoch_features = []

    # 2. FILTER & WELCH PSD
    for ch in range(eeg_centered.shape[0]):
        # Force contiguous array for C++ BrainFlow filter
        channel_data = np.ascontiguousarray(eeg_centered[ch], dtype=np.float64)

        # Apply identical 60Hz Bandstop filter
        DataFilter.perform_bandstop(channel_data, sampling_rate, 58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0)

        # Compute PSD exactly as real-time does
        freqs, psd = scipy.signal.welch(channel_data, fs=sampling_rate, nperseg=sampling_rate)

        # Isolate Alpha band (8-13 Hz)
        alpha_indices = np.logical_and(freqs >= 8.0, freqs <= 13.0)
        alpha_power = np.mean(psd[alpha_indices])

        epoch_features.append(alpha_power)

    features_list.append(epoch_features)

# Convert to final Machine Learning Matrix
X = np.array(features_list)

# Extract the labels (1 for EyesClosed, 2 for EyesOpen)
y = all_epochs.events[:, 2]

print(f"Feature Matrix (X) Shape: {X.shape} -> (Total Epochs, Channels)")
print(f"Labels Vector (y) Shape: {y.shape} -> (Total Epochs)")

# ==========================================
# PHASE 3: MACHINE LEARNING (LDA)
# ==========================================
print("\n--- Training the LDA Classifier ---")

# 1. Train/Test Split (Keep 30% of the data completely hidden to test real-world accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Scale the Features (CRITICAL for real-time inference)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Note: we only transform the test set, we don't fit!

# 3. Initialize and Train the Model using the scaled data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

# 4. Test the Model on the hidden 30%
y_pred = lda.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n>> FINAL MODEL ACCURACY: {accuracy * 100:.2f}% <<\n")

# 5. Cross-Validation (Scale the whole dataset first for this test)
X_scaled_all = scaler.fit_transform(X)
cv_scores = cross_val_score(lda, X_scaled_all, y, cv=5)
print(f"5-Fold Cross-Validation Average: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Eyes Closed (1)', 'Eyes Open (2)']))

# ==========================================
# PHASE 4: SAVE MODEL FOR REAL-TIME USE
# ==========================================
print("\n--- Saving Model to Disk ---")

# Save both the trained model and the scaler
joblib.dump(lda, 'lda_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("SUCCESS: 'lda_model.pkl' and 'scaler.pkl' have been saved to your folder.")
print("You are ready for real-time streaming!")