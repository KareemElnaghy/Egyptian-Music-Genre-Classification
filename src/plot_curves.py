import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

log_path = "demo_model/results/demo_run/train_log.csv"
out_dir = "demo_model/results/demo_run"

df = pd.read_csv(log_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Curves — Demo Run", fontsize=14, fontweight="bold")

# Loss
ax = axes[0]
ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2196F3", linewidth=1.8)
ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#F44336", linewidth=1.8, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

# Accuracy
ax = axes[1]
ax.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc", color="#2196F3", linewidth=1.8)
ax.plot(df["epoch"], df["val_acc"] * 100, label="Val Acc", color="#F44336", linewidth=1.8, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

plt.tight_layout()
save_path = f"{out_dir}/training_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved to {save_path}")
plt.show()
