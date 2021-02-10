import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
import pickle
import numpy as np
import sys

# # TODO plot time correctly
# whiten_losses = [i * torch.tensor(range(10)) / 10 for i in range(5)]
# std_losses = [i * torch.tensor(range(10))**2 / 10 for i in range(5)]

fig, ax = plt.subplots(figsize=(14,7))
fig.tight_layout(pad=10)

def plot_results(path, color, newsize=500, select=None, all_results=False):
    results = pickle.load(open(path, "rb"))
    for i, (label, losses) in enumerate(results.items()):
        xs = np.interp(np.linspace(0, len(losses), newsize), np.arange(len(losses)), losses)
        if not all_results or i == select:
            ax.plot(
                np.arange(len(xs)) / newsize * 10,
                xs, 
                label=f"{label.item():.3f}", 
                c=color, 
                linewidth=1/2,
                alpha=1 if all_results else (i+1) * (1/(3*len(results)) if i != select else 1)
            )

if True:
    title = "Impact of blocksize on optimization speed (learning rate 0.25)"
    sizes = [1, 2, 4, 8, 16, 32, 64]
    colors = ["grey", "black", "cyan", "teal", "purple", "green", "blue"]
    plot_results("work_dirs/optimize_demo_std/results.pkl", "red", select=8, all_results=True)
    for color, size in zip(colors, sizes):
        plot_results(f"work_dirs/optimize_demo_{size}/results.pkl", color, select=8, all_results=True)
    ax.legend(
        ["std"] + sizes,
        loc='upper right', bbox_to_anchor=(1.1, 1), title="block size", 
        )
else:
    title = "Impact of blockwise whitening on optimization: optimization losses over 10 seconds for different learning rates"
    plot_results("work_dirs/optimize_demo_std/results.pkl", "red", select=8)
    plot_results("work_dirs/optimize_demo_64/results.pkl", "blue", select=7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), title="l.r.")


plt.title(title, {'fontsize': 16}, y=1.12)
# plt.suptitle("Red: block size 1, Green: block size 32, Blue: block size 64. Highlighted: best rates.", y=0.85)
plt.xlabel("Seconds")
if False:
    plt.yscale("log")
else:
    plt.ylim(0.0, 0.075)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.ylabel("Loss")
plt.savefig("work_dirs/optim_results.png")