# config.py
# Central configuration file for all augmentation settings

import os

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
AUGMENTED_DATA_DIR = os.path.join(BASE_DIR, "data", "augmented")

# ─────────────────────────────────────────
# AUDIO SETTINGS
# ─────────────────────────────────────────
SAMPLE_RATE = 22050       # Standard sample rate for audio (Hz)
DURATION = 5              # Clip all audio to 5 seconds
AUDIO_FORMAT = ".wav"     # Output format

# ─────────────────────────────────────────
# AUGMENTATION SETTINGS
# ─────────────────────────────────────────
AUGMENTATIONS_PER_FILE = 5   # How many augmented copies per original file
                              # 1,536 x 5 = 7,680 new files

# Classes (your cough types)
CLASSES = ["dry", "wet", "wheezy"]

# ─────────────────────────────────────────
# AUGMENTATION TECHNIQUE SETTINGS
# ─────────────────────────────────────────
NOISE_MIN_AMPLITUDE = 0.001   # Minimum background noise to add
NOISE_MAX_AMPLITUDE = 0.015   # Maximum background noise to add

PITCH_MIN_SEMITONES = -3      # Shift pitch down by max 3 semitones
PITCH_MAX_SEMITONES = 3       # Shift pitch up by max 3 semitones

TIME_STRETCH_MIN = 0.8        # Slow audio down to 80% speed
TIME_STRETCH_MAX = 1.2        # Speed audio up to 120% speed

SHIFT_MIN_FRACTION = -0.2     # Shift audio start left by 20%
SHIFT_MAX_FRACTION = 0.2      # Shift audio start right by 20%