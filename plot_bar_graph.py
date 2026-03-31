import pandas as pd
import matplotlib.pyplot as plt

base_path = "/home/students1/mantani/IEEE_TEMPR_value_stacking/results"

# ===============================
# CSVから最終行の平均を取る関数
# ===============================
def get_row_mean(prefix, timestamp):
    last_rows = []
    col_names = None

    for i in range(1, 11):
        file = f"{base_path}/episode_results_{prefix}_{i}_{timestamp}.csv"

        try:
            df = pd.read_csv(file)
        except FileNotFoundError:
            print(f"not found: {file}")
            continue

        if df.empty:
            print(f"empty file: {file}")
            continue

        last_rows.append(df.iloc[-1].values)

        if col_names is None:
            col_names = df.columns.tolist()

    if len(last_rows) == 0:
        raise ValueError(f"{prefix} のCSVが1つも読めませんでした")

    result = pd.DataFrame(last_rows).T
    result.index = col_names[:result.shape[0]]

    return result.mean(axis=1)


# ===============================
# データ取得
# ===============================
mean_h1 = get_row_mean("hybrid", "20260318_1933")
mean_c1 = get_row_mean("co-located", "20260318_1933")

mean_h2 = get_row_mean("hybrid", "20260318_1942")
mean_c2 = get_row_mean("co-located", "20260318_1942")

means = [mean_h1, mean_c1, mean_h2, mean_c2]

# x軸
x_positions = [0, 1, 3, 4]

x_labels = [
    "Hybrid\n(No Oversizing)",
    "Co-located\n(No Oversizing)",
    "Hybrid\n(Oversizing)",
    "Co-located\n(Oversizing)"
]

labels = mean_h1.index

# ===============================
# 色設定
# ===============================
colors = {
    "revenue_energy": "tab:blue",
    "revenue_fcas": "tab:orange",
    "capacity_payment": "tab:green",

}

# ===============================
# 凡例名（←ここで自由に指定）
# ===============================
legend_names = {
    "revenue_energy": "Energy Revenue",
    "revenue_fcas": "Ancillary Revenue",
    "capacity_payment": "Capacity Payment",
}


# ===============================
# プロット
# ===============================
plt.figure(figsize=(10, 6))

used_labels = set()

for i, mean in enumerate(means):

    bottom_pos = 0
    bottom_neg = 0

    for label in labels:

        # ===============================
        # 除外（表示しない）
        # ===============================
        if label in {"imbalance_penalty", "cost_degradation","prediction_penalty"}:
            continue

        # ===============================
        # imbalanceをenergyに吸収
        # ===============================
        if label == "revenue_energy":
            val = mean["revenue_energy"] - abs(mean["imbalance_penalty"])
        else:
            val = mean[label]

        color = colors.get(label, "gray")
        legend_name = legend_names.get(label, label)

        # 凡例の重複防止
        legend_label = legend_name if legend_name not in used_labels else None
        used_labels.add(legend_name)

        # ===============================
        # 棒グラフ
        # ===============================
        v = abs(val)
        plt.bar(x_positions[i], v, bottom=bottom_pos,
                color=color, label=legend_label)
        bottom_pos += v

    # ===============================
    # 合計値表示
    # ===============================
    plt.text(x_positions[i], bottom_pos,
             f"{bottom_pos:.1f}", ha="center", va="bottom")

# ===============================
# 軸設定
# ===============================
plt.xticks(x_positions, x_labels)
plt.axhline(0)

plt.ylabel("Average Revenue($)")
plt.ylim(0, 80000)

# 凡例
plt.legend(
    bbox_to_anchor=(0.5, 1.1),
    loc="upper center",
    fontsize=12,
    ncol = 4
)

plt.tight_layout()

# 保存
plt.savefig(
    "/home/students1/mantani/IEEE_TEMPR_value_stacking/plot/plot_4bars.pdf"
)

plt.show()