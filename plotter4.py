import matplotlib.pyplot as plt
import numpy as np

# File paths for each model
file_paths = {
    "llama2-13b": "/home/sohamr/projects/federated-learning/flowertune-llm/flowertune-llm/all_losses/medical_qa/llama_2_13b_16bit_82024-10-06_14-27-43/all_losses.txt",
    "llama3.2-3b": "/home/sohamr/projects/federated-learning/flowertune-llm/flowertune-llm/all_losses/medical_qa/llama3_2_3b_16bit_42024-10-06_13-15-14/all_losses.txt",
    "llama3.1-8b": "/home/sohamr/projects/federated-learning/flowertune-llm/flowertune-llm/all_losses/medical_qa/llama_3_1_16bit_82024-10-06_18-01-54/all_losses.txt",
    "mistral-nemo-12b": "/home/sohamr/projects/federated-learning/flowertune-llm/flowertune-llm/all_losses/medical_qa/mistral_16bit_nemo_82024-10-07_01-43-56/all_losses.txt"
}

# Save path
SAVE_PATH = "./comparison_16bit_models.png"

# Read and process loss data


def read_losses(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        return np.array([float(loss) for loss in file.read().splitlines()])


# Prepare plot
fig, ax = plt.subplots(figsize=(12, 7))

# Define colors and markers for each model
colors = ['blue', 'orange', 'green', 'red']
markers = ['o', 's', '^', 'd']

# Plot each model's losses and add lines for lowest losses
for i, (model_name, file_path) in enumerate(file_paths.items()):
    losses = read_losses(file_path)
    ax.plot(losses, color=colors[i], linestyle='-',
            marker=markers[i], label=model_name, markersize=5)

    # Add line for the lowest loss value
    min_loss = np.min(losses)
    ax.axhline(
        y=min_loss, color=colors[i], linestyle='--', label=f'Lowest Loss {model_name}')

# Set title and labels with increased font sizes
ax.set_title('Comparison of 16-bit Model Losses', fontsize=16)
ax.set_xlabel('Training Round', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)

# Customize the legend
ax.legend(fontsize=10, loc='upper right')

# Enable grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits to focus on the range of interest
ax.set_ylim(bottom=min([np.min(read_losses(fp)) for fp in file_paths.values()]) - 0.1,
            top=max([np.max(read_losses(fp)) for fp in file_paths.values()]) + 0.1)

# Improve tick label size
ax.tick_params(axis='both', which='major', labelsize=10)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()