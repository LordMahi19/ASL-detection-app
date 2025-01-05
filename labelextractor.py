import numpy as np

# Load the NPZ file
data = np.load("sign_language_dataset.npz")

# Get the labels array
labels = data['labels']

# Get unique labels in the order they were first seen
unique_labels = []
for label in labels:
    if label not in unique_labels:
        unique_labels.append(label)

# Save to labels.txt
with open('labels.txt', 'w') as f:
    for label in unique_labels:
        f.write(f"{label}\n")

print("Labels extracted and saved to labels.txt")
print("Unique labels:", unique_labels)
print("Total number of unique labels:", len(unique_labels))