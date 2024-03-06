import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the data for the x-axis and y-axis.
x_values = [4, 512, 512, 256, 128, 64]  # X-axis data derived from the first value in each x_data array.
y_data = [
    [0.7968, 0.6667, 0.8201],
    [0.8, 0.7873, 0.7587],
    [0.7968, 0.7873, 0.7492],
    [0.7524, 0.5651, 0.5587],
    [0.6095, 0.4921, 0.4095],
    [0.5365, 0.5397, None]  # Assuming None for the missing value.
]

# Convert y_data to a numpy array while handling missing values.
np_y_data = np.array(y_data, dtype=object)

# Plot each series and a horizontal line at y=0.7877.
plt.figure(figsize=(10, 5))
for i in range(np_y_data.shape[1]):
    column_data = np_y_data[:, i]
    valid_indices = ~pd.isnull(column_data)
    plt.plot(np.array(x_values)[valid_indices], column_data[valid_indices], marker='o', label=f'Series {i+1}')
plt.axhline(y=0.7877, color='r', linestyle='--', label='y = 0.7877')

plt.title('Line Chart with Horizontal Line at y=0.7877')
plt.xlabel('X-axis Information')
plt.ylabel('Value')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()