import numpy as np
import matplotlib.pyplot as plt

data = np.load("sign_language_dataset.npz")
hand_data = data['data']
labels = data['labels']

sample_index = 50500  # Choose the index of the hand you want to visualize
hand_landmarks = hand_data[sample_index].reshape(21, 2)

# Plot the landmarks
plt.figure(figsize=(6, 6))
plt.scatter(hand_landmarks[:, 0], hand_landmarks[:, 1], s=50) # s makes the points bigger

# Connect the landmarks (MediaPipe hand model connections)
connections = [[0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
               [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
               [5, 9], [9, 10], [10, 11], [11, 12], # Middle finger
               [9, 13], [13, 14], [14, 15], [15, 16], # Ring finger
               [13, 17], [17, 18], [18, 19], [19, 20], # Pinky finger
               [0, 17]] # Palm connection
for connection in connections:
    x_values = [hand_landmarks[connection[0], 0], hand_landmarks[connection[1], 0]]
    y_values = [hand_landmarks[connection[0], 1], hand_landmarks[connection[1], 1]]
    plt.plot(x_values, y_values, 'r-', linewidth=2) # linewidth makes lines thicker

# Add landmark numbers (optional but very helpful for debugging)
for i, (x, y) in enumerate(hand_landmarks):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,10), ha='center') # Add numbers

plt.title(f"Hand Landmarks (Label: {labels[sample_index]})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.gca().invert_yaxis()
plt.show()