import matplotlib.pyplot as plt
import numpy as np

# File paths
fileName1_4bit = input("Enter the path to the 4-bit model loss file: ")
fileName2_16bit = input("Enter the path to the 16-bit model loss file: ")

# Save path
SAVE_PATH = input("Enter the path to save the plot: ")

# Read and process loss data


def read_losses(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        return np.array([float(loss) for loss in file.read().splitlines()])


loss_4bit = read_losses(fileName1_4bit)
loss_16bit = read_losses(fileName2_16bit)

# Calculate the lowest loss for both models
min_loss_4bit = np.min(loss_4bit)
min_loss_16bit = np.min(loss_16bit)

# Create a figure and axis with improved size
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the losses with improved styles
ax.plot(loss_4bit, color='blue', linestyle='-',
        marker='o', label='4-bit model', markersize=5)
ax.plot(loss_16bit, color='orange', linestyle='-',
        marker='s', label='16-bit model', markersize=5)

# Add lines for the lowest loss values
ax.axhline(y=min_loss_4bit, color='blue',
           linestyle='--', label='Lowest Loss 4-bit')
ax.axhline(y=min_loss_16bit, color='orange',
           linestyle='--', label='Lowest Loss 16-bit')

# Set title and labels with increased font sizes
ax.set_title('Comparison of Model Losses with Focused Range', fontsize=16)
ax.set_xlabel('Training Round', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)

# Customize the legend
ax.legend(fontsize=10, loc='upper right')

# Enable grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits to focus on the range of interest
ax.set_ylim(bottom=min(min_loss_4bit, min_loss_16bit) - 0.1,
            top=max(max(loss_4bit), max(loss_16bit)) + 0.1)

# Improve tick label size
ax.tick_params(axis='both', which='major', labelsize=10)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
