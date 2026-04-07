# ============================================================
# FILE: data_augmentation.py
# PURPOSE: Audio Data Augmentation for Cough Classification
# AUTHOR: Your Project
# ============================================================

# ============================================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# Run this in PyCharm Terminal first:
# pip install librosa soundfile numpy tqdm
# ============================================================

import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 2: DEFINE PATHS
# ============================================================

INPUT_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\converted_dataset"

OUTPUT_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\augmented_dataset"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"✅ Input  Path : {INPUT_PATH}")
print(f"✅ Output Path : {OUTPUT_PATH}")

# ============================================================
# STEP 3: DEFINE ALL AUGMENTATION FUNCTIONS
# ============================================================

# --- 1. Add Background Noise ---
def add_noise(audio, noise_factor=0.005):
    """Adds slight random noise to the audio"""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented.astype(np.float32)

# --- 2. Pitch Shift Up ---
def pitch_shift_up(audio, sr, n_steps=2):
    """Shifts pitch upward by n_steps semitones"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

# --- 3. Pitch Shift Down ---
def pitch_shift_down(audio, sr, n_steps=-2):
    """Shifts pitch downward by n_steps semitones"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

# --- 4. Time Stretch (Slow Down) ---
def time_stretch(audio, rate=0.8):
    """Slows down the audio without changing pitch"""
    return librosa.effects.time_stretch(audio, rate=rate)

# --- 5. Time Shift ---
def time_shift(audio, sr, shift_max=0.2):
    """Shifts audio forward or backward in time"""
    shift = int(np.random.uniform(-shift_max, shift_max) * sr)
    augmented = np.roll(audio, shift)
    if shift > 0:
        augmented[:shift] = 0
    else:
        augmented[shift:] = 0
    return augmented.astype(np.float32)

# --- 6. Volume Scaling ---
def volume_scale(audio, factor=1.5):
    """Increases or decreases the volume"""
    return (audio * factor).astype(np.float32)

print("✅ All augmentation functions defined successfully!")

# ============================================================
# STEP 4: LOAD AND VALIDATE AUDIO FILES
# ============================================================

# Get all .wav files from input folder
all_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.wav')]
total_files = len(all_files)

print(f"\n📂 Total .wav files found : {total_files}")

if total_files == 0:
    print("❌ No .wav files found! Please check your INPUT_PATH.")
    exit()

# ============================================================
# STEP 5: APPLY AUGMENTATION TO ALL FILES
# ============================================================

print("\n🚀 Starting Data Augmentation...\n")

# Track counts
success_count = 0
error_count = 0
error_files = []

for filename in tqdm(all_files, desc="Augmenting Files", unit="file"):

    file_path = os.path.join(INPUT_PATH, filename)
    base_name = os.path.splitext(filename)[0]  # filename without .wav

    try:
        # --- Load audio file ---
        audio, sr = librosa.load(file_path, sr=None)  # sr=None keeps original sample rate

        # -----------------------------------------------
        # Save Version 1: Original (copy to output folder)
        # -----------------------------------------------
        out_original = os.path.join(OUTPUT_PATH, f"{base_name}_original.wav")
        sf.write(out_original, audio, sr)

        # -----------------------------------------------
        # Save Version 2: Add Noise
        # -----------------------------------------------
        audio_noise = add_noise(audio)
        out_noise = os.path.join(OUTPUT_PATH, f"{base_name}_noise.wav")
        sf.write(out_noise, audio_noise, sr)

        # -----------------------------------------------
        # Save Version 3: Pitch Shift Up
        # -----------------------------------------------
        audio_pitch_up = pitch_shift_up(audio, sr)
        out_pitch_up = os.path.join(OUTPUT_PATH, f"{base_name}_pitch_up.wav")
        sf.write(out_pitch_up, audio_pitch_up, sr)

        # -----------------------------------------------
        # Save Version 4: Pitch Shift Down
        # -----------------------------------------------
        audio_pitch_down = pitch_shift_down(audio, sr)
        out_pitch_down = os.path.join(OUTPUT_PATH, f"{base_name}_pitch_down.wav")
        sf.write(out_pitch_down, audio_pitch_down, sr)

        # -----------------------------------------------
        # Save Version 5: Time Stretch
        # -----------------------------------------------
        audio_time_stretch = time_stretch(audio)
        out_time_stretch = os.path.join(OUTPUT_PATH, f"{base_name}_time_stretch.wav")
        sf.write(out_time_stretch, audio_time_stretch, sr)

        # -----------------------------------------------
        # Save Version 6: Time Shift
        # -----------------------------------------------
        audio_time_shift = time_shift(audio, sr)
        out_time_shift = os.path.join(OUTPUT_PATH, f"{base_name}_time_shift.wav")
        sf.write(out_time_shift, audio_time_shift, sr)

        # -----------------------------------------------
        # Save Version 7: Volume Scale
        # -----------------------------------------------
        audio_volume = volume_scale(audio)
        out_volume = os.path.join(OUTPUT_PATH, f"{base_name}_volume.wav")
        sf.write(out_volume, audio_volume, sr)

        success_count += 1

    except Exception as e:
        error_count += 1
        error_files.append(filename)
        print(f"\n⚠️  Error processing {filename}: {str(e)}")

# ============================================================
# STEP 6: FINAL SUMMARY REPORT
# ============================================================

print("\n" + "="*55)
print("           ✅ AUGMENTATION COMPLETE!")
print("="*55)
print(f"  Original Files         : {total_files}")
print(f"  Successfully Processed : {success_count}")
print(f"  Failed Files           : {error_count}")
print(f"  Augmentation Versions  : 7 per file")
print(f"  Total Output Files     : {success_count * 7}")
print(f"  Output Saved To        : {OUTPUT_PATH}")
print("="*55)

if error_files:
    print(f"\n⚠️  Files with errors:")
    for ef in error_files:
        print(f"   - {ef}")

print("\n🎯 Next Step: Convert augmented audio to Spectrograms!")