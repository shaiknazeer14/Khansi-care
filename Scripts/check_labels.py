# ============================================================
# FILE: check_labels.py
# PURPOSE: Diagnose label mapping issue
# ============================================================

import os
import json

# Paths
JSON_PATH  = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\converted_dataset"
AUDIO_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\augmented_dataset"

# -----------------------------------------------
# Step 1: Show sample JSON filenames
# -----------------------------------------------
json_files = [f for f in os.listdir(JSON_PATH) if f.endswith('.json')]
wav_files  = [f for f in os.listdir(JSON_PATH) if f.endswith('.wav')]
aug_files  = [f for f in os.listdir(AUDIO_PATH) if f.endswith('.wav')]

print("="*60)
print("  SAMPLE JSON FILES (first 5):")
print("="*60)
for f in json_files[:5]:
    print(f"  {f}")

print("\n" + "="*60)
print("  SAMPLE WAV FILES IN ORIGINAL FOLDER (first 5):")
print("="*60)
for f in wav_files[:5]:
    print(f"  {f}")

print("\n" + "="*60)
print("  SAMPLE WAV FILES IN AUGMENTED FOLDER (first 5):")
print("="*60)
for f in aug_files[:5]:
    print(f"  {f}")

# -----------------------------------------------
# Step 2: Check if JSON UUID matches WAV UUID
# -----------------------------------------------
print("\n" + "="*60)
print("  CHECKING UUID MATCH BETWEEN JSON & AUDIO:")
print("="*60)

# Get first JSON file and read it
if json_files:
    sample_json = json_files[0]
    sample_uuid = os.path.splitext(sample_json)[0]
    json_path   = os.path.join(JSON_PATH, sample_json)

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"\n  Sample JSON filename : {sample_json}")
    print(f"  Sample UUID          : {sample_uuid}")
    print(f"  cough_detected value : {data.get('cough_detected', 'NOT FOUND')}")
    print(f"  status field         : {data.get('status', 'NOT FOUND')}")

    # Check if matching wav exists
    matching_wav = sample_uuid + '.wav'
    wav_exists   = os.path.exists(os.path.join(JSON_PATH, matching_wav))
    print(f"\n  Looking for WAV      : {matching_wav}")
    print(f"  WAV exists in folder : {wav_exists}")

# -----------------------------------------------
# Step 3: Check cough_detected score distribution
# -----------------------------------------------
print("\n" + "="*60)
print("  COUGH_DETECTED SCORE DISTRIBUTION:")
print("="*60)

dry_count    = 0
wet_count    = 0
wheezy_count = 0
missing      = 0

for jf in json_files:
    try:
        with open(os.path.join(JSON_PATH, jf), 'r') as f:
            data = json.load(f)
        score = float(data.get('cough_detected', 0))
        if score <= 0.3:
            dry_count += 1
        elif score <= 0.7:
            wet_count += 1
        else:
            wheezy_count += 1
    except:
        missing += 1

print(f"  🟢 Dry    (0.0-0.3) : {dry_count}")
print(f"  🟡 Wet    (0.3-0.7) : {wet_count}")
print(f"  🔴 Wheezy (0.7-1.0) : {wheezy_count}")
print(f"  ❌ Missing/Error    : {missing}")
print(f"  📂 Total JSON files : {len(json_files)}")

print("\n🎯 Share this output to fix the labeling issue!")