import matplotlib.pyplot as plt

FOLDER = '/home/sohamr/sohamr/fed-llm-fintuning-and-blockchain/results/models--openai-community--gpt242024-10-04_08-46-48'

FILE_NAME='/gpt2-drug_bank-8_3_clients.png'

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