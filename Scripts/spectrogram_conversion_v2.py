# ============================================================
# FILE: spectrogram_conversion_v2.py
# PURPOSE: Fixed version - correct JSON path
# ============================================================

import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS - FIXED
# ============================================================

# ✅ FIXED: Correct JSON path
JSON_PATH  = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\dataset"

# Augmented audio files
AUDIO_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\augmented_dataset"

# Output spectrogram folder
OUTPUT_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\spectrograms"

# Create output subfolders
for label in ['dry', 'wet', 'wheezy']:
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

print(f"✅ JSON Path    : {JSON_PATH}")
print(f"✅ Audio Path   : {AUDIO_PATH}")
print(f"✅ Output Path  : {OUTPUT_PATH}")

# ============================================================
# AUTO LABELING FUNCTION
# ============================================================

def get_label_from_cough_score(cough_score):
    """
    0.0 - 0.3  →  dry
    0.3 - 0.7  →  wet
    0.7 - 1.0  →  wheezy
    """
    try:
        score = float(cough_score)
        if score <= 0.3:
            return 'dry'
        elif score <= 0.7:
            return 'wet'
        else:
            return 'wheezy'
    except:
        return 'dry'

# ============================================================
# BUILD LABEL MAP FROM JSON FILES
# ============================================================

print("\n📂 Reading JSON files to build label map...")

label_map  = {}
json_files = [f for f in os.listdir(JSON_PATH) if f.endswith('.json')]

print(f"   Total JSON files found: {len(json_files)}")

dry_count    = 0
wet_count    = 0
wheezy_count = 0

for json_file in tqdm(json_files, desc="Reading JSON Labels", unit="file"):
    json_file_path = os.path.join(JSON_PATH, json_file)
    base_name      = os.path.splitext(json_file)[0]  # UUID without .json

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        cough_score = data.get('cough_detected', '0.0')
        label       = get_label_from_cough_score(cough_score)
        label_map[base_name] = label

        if label == 'dry':    dry_count    += 1
        elif label == 'wet':  wet_count    += 1
        else:                 wheezy_count += 1

    except Exception as e:
        label_map[base_name] = 'dry'
        dry_count += 1

print(f"\n✅ Label Map Built!")
print(f"   Total Mapped     : {len(label_map)}")
print(f"   🟢 Dry    (0.0-0.3) : {dry_count}")
print(f"   🟡 Wet    (0.3-0.7) : {wet_count}")
print(f"   🔴 Wheezy (0.7-1.0) : {wheezy_count}")

# ============================================================
# VERIFY UUID MATCHING
# ============================================================

print("\n🔍 Verifying UUID matching...")

aug_files   = [f for f in os.listdir(AUDIO_PATH) if f.endswith('.wav')]
matched     = 0
not_matched = 0

suffixes = ['_original', '_noise', '_pitch_up',
            '_pitch_down', '_time_stretch',
            '_time_shift', '_volume']

for wav_file in aug_files[:20]:  # Check first 20 files
    base_name = wav_file.replace('.wav', '')
    uuid_name = base_name
    for suffix in suffixes:
        if base_name.endswith(suffix):
            uuid_name = base_name[:-len(suffix)]
            break
    if uuid_name in label_map:
        matched += 1
    else:
        not_matched += 1

print(f"   Matched UUIDs     : {matched}/20")
print(f"   Not Matched UUIDs : {not_matched}/20")

if not_matched > 0:
    print("   ⚠️  Some UUIDs not matching - check paths!")
else:
    print("   ✅ All UUIDs matching perfectly!")

# ============================================================
# MEL SPECTROGRAM FUNCTION
# ============================================================

def save_mel_spectrogram(audio_path, output_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    mel_spec  = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=128,
        fmax=8000,
        n_fft=2048,
        hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(2.24, 2.24))
    plt.axes([0, 0, 1, 1])
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=512,
        x_axis=None,
        y_axis=None,
        cmap='viridis'
    )
    plt.axis('off')
    plt.savefig(output_path, dpi=100,
                bbox_inches='tight', pad_inches=0)
    plt.close()

# ============================================================
# CONVERT ALL AUGMENTED AUDIO TO SPECTROGRAMS
# ============================================================

print(f"\n🚀 Starting Spectrogram Conversion...")
print(f"   Total files to convert: {len(aug_files)}\n")

success_count = 0
error_count   = 0
error_files   = []
label_counts  = {'dry': 0, 'wet': 0, 'wheezy': 0}

for wav_file in tqdm(aug_files, desc="Converting", unit="file"):

    wav_path  = os.path.join(AUDIO_PATH, wav_file)
    base_name = wav_file.replace('.wav', '')

    # Extract UUID by removing augmentation suffix
    uuid_name = base_name
    for suffix in suffixes:
        if base_name.endswith(suffix):
            uuid_name = base_name[:-len(suffix)]
            break

    try:
        # Get label from map
        label = label_map.get(uuid_name, 'dry')

        # Output path
        output_filename = base_name + '.png'
        output_file     = os.path.join(OUTPUT_PATH, label, output_filename)

        # Convert and save
        save_mel_spectrogram(wav_path, output_file)

        label_counts[label] += 1
        success_count += 1

    except Exception as e:
        error_count += 1
        error_files.append(wav_file)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*55)
print("       ✅ SPECTROGRAM CONVERSION COMPLETE!")
print("="*55)
print(f"  Total Audio Files       : {len(aug_files)}")
print(f"  Successfully Converted  : {success_count}")
print(f"  Failed Files            : {error_count}")
print(f"\n  📊 Spectrograms by Label:")
print(f"     🟢 Dry    : {label_counts['dry']}")
print(f"     🟡 Wet    : {label_counts['wet']}")
print(f"     🔴 Wheezy : {label_counts['wheezy']}")
print(f"\n  Output Saved To : {OUTPUT_PATH}")
print("="*55)
print("\n🎯 Next Step: Model Training!")