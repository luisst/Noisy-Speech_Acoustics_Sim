from pathlib import Path
from collections import defaultdict
import pprint
import math
import sys
import random

# Parameters
root_path = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'TTS4_dvec')
folder_path = root_path / "TTS4_input_wavs"
threshold_samples = 40
balanced_samples = 200

# Step 1: Count speaker_ID occurrences
speaker_counts = defaultdict(int)

for wav_file in folder_path.glob("*.wav"):
    id_part = wav_file.stem.split('_')[1]
    if id_part.startswith("ID-") and len(id_part) == 5:
        speaker_counts[id_part] += 1 
    else:
        sys.exit(f"Invalid speaker ID format in file: {wav_file.stem}")


# Print the counts of each speaker ID sorted by count
sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
print("Speaker ID counts:")
for speaker_id, count in sorted_speakers:
    print(f"{speaker_id}: {count}")


# Step 2: Filter speakers with enough samples
eligible_speakers = {
    speaker_id for speaker_id, count in speaker_counts.items()
    if count >= threshold_samples
}

# Step 3: Create dictionary with speaker_ID -> list of wav paths
speaker_files = defaultdict(list)

for wav_file in folder_path.glob("*.wav"):
    id_part = wav_file.stem.split('_')[1]
    if id_part in eligible_speakers:
        speaker_files[id_part].append(wav_file)

# Step 4: Balance the dataset
balanced_dict = dict()

for speaker_id, files in speaker_files.items():
    if len(files) >= balanced_samples:
        # Randomly select balanced_samples from files
        balanced_dict[speaker_id] = random.sample(files, balanced_samples)

    elif len(files) > threshold_samples:
        balanced_dict[speaker_id] = files

# Print the balanced dictionary keys and number of samples
print("\nBalanced Speaker Dictionary:")
for speaker_id, files in balanced_dict.items():
    print(f"{speaker_id}: {len(files)} samples")

# Copy each balanced file to new folder
output_folder = folder_path.parent / "TTS4_input_wavs_balanced"
output_folder.mkdir(exist_ok=True)

for speaker_id, files in balanced_dict.items():
    for wav_path in files:
        new_path = output_folder / wav_path.name
        wav_path.rename(new_path)
