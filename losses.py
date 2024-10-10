import re

# Read the logs from the file
with open('training_logs.txt', 'r') as file:
    logs = file.read()

# Regular expression to match loss values
pattern = r"'loss': (\d+\.\d+)"

# Find all loss values
loss_values = re.findall(pattern, logs)

# Write loss values to a new file
with open('loss_values.txt', 'w') as f:
    for loss in loss_values:
        f.write(f"{loss}\n")

print("Loss values have been written to 'loss_values.txt'")
