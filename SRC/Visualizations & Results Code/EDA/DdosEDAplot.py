import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join("Visualizations & Results", "DDoS", "EDA", SCRIPT_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

X = pd.read_csv("Data/Processed/Botnet_X.csv").drop(columns=["Unnamed: 0"], errors="ignore")
Y = pd.read_csv("Data/Processed/Botnet_Y.csv")

# ── 1. FEATURE DISTRIBUTIONS ─────────────────────────────────────────────────
NUM_FEATURES = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Pkt Len Mean",
    "Pkt Len Std", "Active Mean", "Idle Mean", "Flow IAT Mean"
]
NUM_FEATURES = [c for c in NUM_FEATURES if c in X.columns]

fig, axes = plt.subplots(len(NUM_FEATURES), 2, figsize=(14, 3.5 * len(NUM_FEATURES)))
fig.suptitle("Botnet Dataset – Feature Distributions", fontsize=16, fontweight="bold")

for i, feat in enumerate(NUM_FEATURES):
    data = X[feat].replace([np.inf, -np.inf], np.nan).dropna()
    axes[i, 0].hist(data, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    axes[i, 0].set_title(f"{feat} – Histogram", fontsize=10)
    axes[i, 0].set_xlabel(feat); axes[i, 0].set_ylabel("Count")
    axes[i, 1].boxplot(data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                       medianprops=dict(color="red", linewidth=2))
    axes[i, 1].set_title(f"{feat} – Boxplot", fontsize=10)
    axes[i, 1].set_xlabel(feat)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "DDoS_Feature_Distributions.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Saved: Botnet_Feature_Distributions.png")

# ── 2. CORRELATION HEATMAP ────────────────────────────────────────────────────
numeric_X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
top_cols = numeric_X.var().nlargest(20).index.tolist()
corr = numeric_X[top_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Botnet Dataset – Correlation Heatmap (Top-20 Variance Features)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "DDoS_Correlation_Heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Saved: Botnet_Correlation_Heatmap.png")

# ── 3. CLASS IMBALANCE ────────────────────────────────────────────────────────
counts = Y["Label"].value_counts().sort_index()
labels = ["Normal (0)", "Botnet (1)"]
colors = ["#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Botnet Dataset – Class Imbalance", fontsize=14, fontweight="bold")

bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{val:,}", ha="center", va="bottom", fontweight="bold")
axes[0].set_ylabel("Sample Count"); axes[0].set_title("Class Counts")

axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[1].set_title("Class Proportions")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "DDoS_Class_Imbalance.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Saved: Botnet_Class_Imbalance.png")

# ── 4. MISSING VALUES ANALYSIS ────────────────────────────────────────────────
full = X.copy()
full["Label"] = Y["Label"].values

missing = full.isnull().sum()
missing_pct = (missing / len(full) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Botnet Dataset – Missing Values Analysis", fontsize=14, fontweight="bold")

if missing_df.empty:
    axes[0].text(0.5, 0.5, "No missing values found!", ha="center", va="center",
                 fontsize=14, color="green", transform=axes[0].transAxes)
    axes[0].set_title("Missing Value Counts")
    axes[1].text(0.5, 0.5, "No missing values found!", ha="center", va="center",
                 fontsize=14, color="green", transform=axes[1].transAxes)
    axes[1].set_title("Missing Value Heatmap")
else:
    missing_df["Missing Count"].plot(kind="bar", ax=axes[0], color="#e67e22", edgecolor="white")
    axes[0].set_title("Missing Value Counts per Feature")
    axes[0].set_ylabel("Count"); axes[0].tick_params(axis="x", rotation=45)
    sns.heatmap(full.isnull(), cbar=False, cmap="YlOrRd", ax=axes[1], yticklabels=False)
    axes[1].set_title("Missing Value Heatmap (rows × features)")

# Summary table printed to console
print("\nMissing Values Summary:")
print(f"  Total cells      : {full.size:,}")
print(f"  Missing cells    : {full.isnull().sum().sum():,}")
print(f"  Complete rows    : {full.dropna().shape[0]:,} / {len(full):,}")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "DDoS_Missing_Values.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Saved: Botnet_Missing_Values.png")
