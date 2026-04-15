# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "./codesign_ddpg_logs"

# ===============================
# 読み込みたいログフォルダを直接指定
# ===============================
log_dirs_groups = {
    "Case 1": [os.path.join(base_dir, f"run_0414_1648_hybrid_PV_ratio_1.5_seed_{i}") for i in [1,2,3,5,6,7]],
    "Case 2": [os.path.join(base_dir, f"run_0410_0948_hybrid_PV_ratio_1.5_seed_{i}") for i in [1,2,3,4,5,6,7,8,9,10]],
    "Case 3": [os.path.join(base_dir, f"run_0410_1619_hybrid_PV_ratio_1.5_seed_{i}") for i in [1,2,4,5,6,7,8,9,10]],
    "Case 4": [os.path.join(base_dir, f"run_0414_1111_hybrid_PV_ratio_1.5_seed_{i}") for i in [2,4,7,9,10]],
}

# ===============================
# 各ケースについて，指定フォルダ群から最終行平均を取る関数
# ===============================
def get_row_mean_from_dirs(log_dirs):
    last_rows = []
    col_names = None

    for log_dir in log_dirs:
        # フォルダ内の episode_results_*.csv を探す
        csv_candidates = sorted(glob.glob(os.path.join(log_dir, "episode_results_*.csv")))

        if len(csv_candidates) == 0:
            print(f"csv not found in: {log_dir}")
            continue

        # 通常は1個だけの想定。複数あれば最初のものを使う
        file = csv_candidates[0]

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"failed to read {file}: {e}")
            continue

        if df.empty:
            print(f"empty file: {file}")
            continue

        last_rows.append(df.iloc[-1].values)

        if col_names is None:
            col_names = df.columns.tolist()

    if len(last_rows) == 0:
        raise ValueError("CSVが1つも読めませんでした")

    result = pd.DataFrame(last_rows).T
    result.index = col_names[:result.shape[0]]

    return result.mean(axis=1)


# ===============================
# データ取得
# ===============================
means = []
x_labels = list(log_dirs_groups.keys())

for case_name, log_dirs in log_dirs_groups.items():
    mean_case = get_row_mean_from_dirs(log_dirs)
    means.append(mean_case)

# x軸
x_positions = list(range(len(x_labels)))
#labels = means[0].index
labels = [
    "revenue_energy",
    "revenue_fcas",
    "capacity_payment",
    "imbalance_penalty",
    "cost_degradation",
]


# ===============================
# 色設定
# ===============================
colors = {
    "revenue_energy": "tab:green",
    "revenue_fcas": "tab:orange",
    "capacity_payment": "tab:blue",
    "imbalance_penalty": "tab:purple",
    "cost_degradation": "tab:red",
}

# ===============================
# 凡例名
# ===============================
legend_names = {
    "revenue_energy": "Energy Revenue",
    "revenue_fcas": "Ancillary Revenue",
    "capacity_payment": "Capacity Payment",
    "imbalance_penalty": "Imbalance Penalty",
    "cost_degradation": "Degradation Cost",
}

# ===============================
# プロット
# ===============================
plt.figure(figsize=(7, 5))

used_labels = set()

for i, mean in enumerate(means):

    bottom_pos = 0
    bottom_neg = 0

    for label in labels:

        if label in {"prediction_penalty"}:
            continue

        val = mean[label]
        color = colors.get(label, "gray")
        legend_name = legend_names.get(label, label)

        if legend_name not in used_labels:
            legend_label = legend_name
            used_labels.add(legend_name)
        else:
            legend_label = None

        if label in {"imbalance_penalty", "cost_degradation"}:
            v = -abs(val)
            plt.bar(
                x_positions[i],
                v,
                bottom=bottom_neg,
                color=color,
                label=legend_label,
                zorder=3
            )
            bottom_neg += v
        else:
            v = abs(val)
            plt.bar(
                x_positions[i],
                v,
                bottom=bottom_pos,
                color=color,
                label=legend_label,
                zorder=3
            )
            bottom_pos += v

    total = bottom_pos + bottom_neg

    plt.text(
        x_positions[i],
        bottom_pos,
        f"{total:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=4
    )

# ===============================
# 軸設定
# ===============================
plt.xticks(x_positions, x_labels, fontsize=12)
plt.axhline(0, color="black", linewidth=1, zorder=2)
plt.ylabel("Average Revenue ($)", fontsize=12)

plt.legend(
    bbox_to_anchor=(0.5, 1.15),
    loc="upper center",
    fontsize=11,
    ncol=3
)

plt.tight_layout()

# 保存
os.makedirs("./plot", exist_ok=True)
plt.savefig("./plot/plot_bars_graph_4cases.pdf")
plt.show()

# ===============================
# means を表形式で整形して表示
# ===============================
summary_df = pd.DataFrame(means, index=x_labels).T

rename_map = {
    "revenue_energy": "Energy Revenue",
    "revenue_fcas": "Ancillary Revenue",
    "capacity_payment": "Capacity Payment",
    "cost_degradation": "Degradation Cost",
    "imbalance_penalty": "Imbalance Penalty",
    "prediction_penalty": "Prediction Penalty",
}
summary_df.rename(index=rename_map, inplace=True)

print("\n=== Mean values for each case ===")
print(summary_df.round(2).to_string())


print("\n=== Annualized Revenues ===")
summary_df.loc['Energy Market'] = (
    summary_df.loc['Energy Revenue'] - summary_df.loc['Imbalance Penalty']
)
print(summary_df * 365/7 )

