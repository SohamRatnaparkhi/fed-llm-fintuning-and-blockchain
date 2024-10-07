import matplotlib.pyplot as plt

FOLDER = '/home/sohamr/sohamr/fed-llm-fintuning-and-blockchain/results/llama_3_1_70b_42024-10-06_00-35-45'

FILE_NAME='/llama_3_1_70b_4_4_bit.png'

LOSS_FILE = FOLDER + '/all_losses.txt'

losses = open(LOSS_FILE).read().splitlines()

losses = [float(loss) for loss in losses]
x = list(range(1, len(losses) + 1))

print(losses)

plt.figure(figsize=(12, 6))
plt.plot(x, losses)
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Losses over Rounds")
plt.ylim(min(losses) - 0.05, max(losses) + 0.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

SAVE_PATH = FOLDER + FILE_NAME
print(SAVE_PATH)
plt.savefig(SAVE_PATH)