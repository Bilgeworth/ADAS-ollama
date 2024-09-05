import json
import os
import pickle
import random
from collections import defaultdict

# Directory containing the benchmark data
data_dir = "_generalized/benchmark_data"
data_files = []

# Get the list of files in the directory
for filename in os.listdir(data_dir):
    if os.path.isfile(os.path.join(data_dir, filename)):
        data_files.append(filename)

# List to store eligible data entries
eligible_entries = []

# Process each file
for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as file:
        data_data = json.load(file)

    # Check if data pair list exists and 'input' exist, if so, add to eligible entries
    if 'data_library' in data_data and len(data_data['data_library']) > 0 and 'input' in data_data['data_library'][0]:
        eligible_entries.append(data_data)

# Randomly sample 100 entries
sample_size = min(100, len(eligible_entries))
print(f"Sampling {sample_size} entries from {len(eligible_entries)} eligible entries")
sampled_entries = random.sample(eligible_entries, sample_size)

# Split the sampled entries into validation and test sets
half_size = 20
validation_entries = sampled_entries[:20]
test_entries = sampled_entries[20:]

# Save validation entries to a pickle file
with open('_generalized/sampled_data_val_data.pkl', 'wb') as val_file:
    pickle.dump(validation_entries, val_file)

# Save test entries to a pickle file
with open('_generalized/sampled_data_test_data.pkl', 'wb') as test_file:
    pickle.dump(test_entries, test_file)

# Calculate and print length statistics for the validation set
val_length_counts = defaultdict(int)
for entry in validation_entries:
    length = len(entry['data_library'][0]['input'])
    val_length_counts[length] += 1

# Output length stats for the validation set
print("Validation Set Length Stats:")
for length, count in sorted(val_length_counts.items()):
    print(f"Length: {length}, Count: {count}")

# Calculate and print length statistics for the test set
test_length_counts = defaultdict(int)
for entry in test_entries:
    length = len(entry['data_library'][0]['input'])
    test_length_counts[length] += 1

# Output length stats for the test set
print("Test Set Length Stats:")
for length, count in sorted(test_length_counts.items()):
    print(f"Length: {length}, Count: {count}")