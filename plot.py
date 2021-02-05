import matplotlib.pyplot as plt
import pickle



# # TODO plot time correctly
# whiten_losses = [i * torch.tensor(range(10)) / 10 for i in range(5)]
# std_losses = [i * torch.tensor(range(10))**2 / 10 for i in range(5)]

fig, ax = plt.subplots(figsize=(14,7))
fig.tight_layout(pad=10)

def plot_results(path, color, select=None):
    results = pickle.load(open(path, "rb"))
    for i, (label, losses) in enumerate(results.items()):
        ax.plot(losses, label=f"{label.item():.3f}", c=color, alpha=(i+1)*(1/(3*len(results)) if i != select else 1))

plot_results("work_dirs/optimize_block/results.pkl", "blue", select=7)
plot_results("work_dirs/optimize_std/results.pkl", "red", select=8)

plt.title("Optimization loss across iterations for different learning rates", {'fontsize': 20}, y=1.12)
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
plt.suptitle("Interrupted after 10 seconds of optimization", y=0.85)
# plt.xlabel("Iterations")
plt.yscale("log")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# plt.ylabel("Loss")
plt.savefig("work_dirs/optim_results.png")