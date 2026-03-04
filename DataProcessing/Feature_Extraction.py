import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# PHASE 1: LOAD, ALIGN, AND COMBINE DATA
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

# 2. ALIGNMENT FIX: Find the common channels across all loaded files
if len(epochs_list) > 1:
    print("\nAligning channels across multiple files...")
    # Get the channel names of the first file to start our comparison
    common_channels = set(epochs_list[0].ch_names)
    
    # Intersect with the rest of the files to find the common denominator
    for epo in epochs_list[1:]:
        common_channels = common_channels.intersection(set(epo.ch_names))
    
    common_channels = list(common_channels)
    print(f"Channels shared by all files: {common_channels}")
    
    # Force every file to only use these shared channels
    for i in range(len(epochs_list)):
        epochs_list[i].pick(common_channels)

# 3. Combine all the aligned files into one massive dataset
all_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
print(f"\nTotal epochs loaded across all files: {len(all_epochs)}")

# ==========================================
# PHASE 2: FEATURE EXTRACTION (The Alpha Band)
# ==========================================
print("\n--- Extracting Alpha Power Features ---")

# Calculate Power Spectral Density specifically for the 8-13 Hz Alpha band
alpha_spectrum = all_epochs.compute_psd(method='welch', fmin=8.0, fmax=13.0, verbose=False)
psds = alpha_spectrum.get_data()

# Average the power across the frequency bins to get one number per channel
X = np.mean(psds, axis=2)

# Extract the labels (1 for EyesClosed, 2 for EyesOpen)
y = all_epochs.events[:, 2]

print(f"Feature Matrix (X) Shape: {X.shape} -> (Total Epochs, Channels)")
print(f"Labels Vector (y) Shape: {y.shape} -> (Total Epochs)")

# ==========================================
# PHASE 3: MACHINE LEARNING (LDA)
# ==========================================
print("\n--- Training the LDA Classifier ---")

# 1. Train/Test Split (Keep 20% of the data completely hidden to test real-world accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize and Train the Model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 3. Test the Model on the hidden 20%
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n>> FINAL MODEL ACCURACY: {accuracy * 100:.2f}% <<\n")

# 4. Cross-Validation (A stricter test to ensure the model isn't just getting lucky)
cv_scores = cross_val_score(lda, X, y, cv=5)
print(f"5-Fold Cross-Validation Average: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Eyes Closed (1)', 'Eyes Open (2)']))