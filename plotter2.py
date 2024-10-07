import matplotlib.pyplot as plt
import numpy as np

# Assuming your loss arrays are named 'loss_4bit' and 'loss_16bit'
fileName1_4bit = "/home/sohamr/projects/def-ssamet-ab/sohamr/results/alpaca-gpt4/llama_2_13b_4bit_42024-10-07_11-08-56/all_losses.txt"
fileName2_16bit = "/home/sohamr/projects/def-ssamet-ab/sohamr/results/alpaca-gpt4/llama_2_13b_16bit_42024-10-07_13-24-54/all_losses.txt"

losses1 = open(fileName1_4bit).read().splitlines()
losses1 = [float(loss) for loss in losses1]


losses2 = open(fileName2_16bit).read().splitlines()
losses2 = [float(loss) for loss in losses2]


loss_4bit = np.array(losses1)  # 50 values
loss_16bit = np.array(losses2)  # 50 values

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the losses
ax.plot(loss_4bit, label='4-bit model')
ax.plot(loss_16bit, label='16-bit model')

# Set title and labels
ax.set_title('Model Losses')
ax.set_xlabel('Round')
ax.set_ylabel('Loss')

# Add legend
ax.legend()

# Show the plot
plt.show()


SAVE_PATH = "./img.png"

plt.savefig(SAVE_PATH)