import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# File paths
fileName1_4bit = input("Enter the path to the model with FL loss file: ")
fileName2_16bit = input("Enter the path to the model without FL loss file: ")

# Save path
SAVE_PATH = input("Enter the path to save the plot: ")

# Read and process loss data


def read_losses(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        return np.array([float(loss) for loss in file.read().splitlines()])


loss_4bit = read_losses(fileName1_4bit)
loss_16bit = read_losses(fileName2_16bit)

min_loss_4bit = np.min(loss_4bit)
min_loss_16bit = np.min(loss_16bit)
mean_loss_4bit = np.mean(loss_4bit)
mean_loss_16bit = np.mean(loss_16bit)
var_loss_4bit = np.var(loss_4bit)
var_loss_16bit = np.var(loss_16bit)
std_loss_4bit = np.std(loss_4bit)
std_loss_16bit = np.std(loss_16bit)

# Create a figure and axis with improved size
fig, ax = plt.subplots(figsize=(14, 8))

# Plot the losses with improved styles
# Define colors
blue = 'blue'
orange = 'orange'
light_blue = mcolors.to_rgba('dodgerblue', alpha=0.7)
light_orange = mcolors.to_rgba('coral', alpha=0.7)

# Plot the losses with improved styles
ax.plot(loss_4bit, color=blue, linestyle='-',
        marker='o', label='FL model', markersize=5)
ax.plot(loss_16bit, color=orange, linestyle='-',
        marker='s', label='Without FL model', markersize=5)

# Add lines for the minimum and mean loss values
ax.axhline(y=min_loss_4bit, color=blue, linestyle='--', label='Min Loss FL')
ax.axhline(y=min_loss_16bit, color=orange,
           linestyle='--', label='Min Loss NoFL')
ax.axhline(y=mean_loss_4bit, color=light_blue,
           linestyle='-', linewidth=2, label='Mean Loss FL')
ax.axhline(y=mean_loss_16bit, color=light_orange,
           linestyle='-', linewidth=2, label='Mean Loss NoFL')

# Plot standard deviation ranges
ax.fill_between(range(len(loss_4bit)),
                mean_loss_4bit - std_loss_4bit,
                mean_loss_4bit + std_loss_4bit,
                color=blue, alpha=0.2, label='Std Dev FL')
ax.fill_between(range(len(loss_16bit)),
                mean_loss_16bit - std_loss_16bit,
                mean_loss_16bit + std_loss_16bit,
                color=orange, alpha=0.2, label='Std Dev NoFL')

# Set title and labels with increased font sizes
ax.set_title(
    'Comparison of Model Losses with Statistical Parameters', fontsize=16)
ax.set_xlabel('Training Round', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)

# Customize the legend
ax.legend(fontsize=10, loc='upper right', ncol=2)

# Enable grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits to focus on the range of interest
y_min = min(min(loss_4bit), min(loss_16bit)) - 0.2
y_max = max(max(loss_4bit), max(loss_16bit)) + 0.2
ax.set_ylim(bottom=y_min, top=y_max)

# Improve tick label size
ax.tick_params(axis='both', which='major', labelsize=10)

# Add text annotations for variance and minimum loss
ax.text(0.02, 0.98, f'Var FL: {var_loss_4bit:.4f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=10, color=blue)
ax.text(0.02, 0.94, f'Var NoFL: {var_loss_16bit:.4f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=10, color=orange)
ax.text(0.02, 0.90, f'Min Loss FL: {min_loss_4bit:.4f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=10, color=blue)
ax.text(0.02, 0.86, f'Min Loss NoFL: {min_loss_16bit:.4f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=10, color=orange)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
