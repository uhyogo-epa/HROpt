# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

base_path = "./comparison_logs"

# ===============================
# CSVから最終行の平均を取る関数
# ===============================
def get_row_mean(prefix, timestamp):
    last_rows = []
    col_names = None

    for i in range(1, 1+10):

        file = base_path + f"/run_{timestamp}_{prefix}_seed_{i}/episode_results_{i}.csv"

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
    
    # Get largest revenue
    result = result[result.sum().nlargest(1).index]

    return result.mean(axis=1)


# ===============================
# データ取得
# ===============================

datetime = "0409_1757"
mean_h1 = get_row_mean("hybrid_PV_ratio_2.0", datetime)
mean_c1 = get_row_mean("co-located_PV_ratio_2.0", datetime)

datetime = "0409_1756"
mean_h1 = get_row_mean("hybrid_PV_ratio_1.5", datetime)
mean_c1 = get_row_mean("co-located_PV_ratio_1.5", datetime)

datetime = "0409_1757"
mean_h1 = get_row_mean("hybrid_PV_ratio_1.1", datetime)
mean_c1 = get_row_mean("co-located_PV_ratio_1.1", datetime)


means = [mean_h1, mean_c1]

print(pd.DataFrame({
    "Hybrid": mean_h1,
    "Co-located": mean_c1
}).round(2))


# x軸
x_positions = [0, 1]

x_labels = [
    "Hybrid",
    "Co-located",
]

#labels = mean_h1.index
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
# 今回は2本の棒グラフなので幅を少しスリムにしています
plt.figure(figsize=(8, 6))

used_labels = set()

for i, mean in enumerate(means):

    bottom_pos = 0  # 上向きスタック
    bottom_neg = 0  # 下向きスタック

    for label in labels:

        # ===============================
        # 除外
        # ===============================
        if label in {"prediction_penalty"}:
            continue

        val = mean[label]
        color = colors.get(label, "gray")
        legend_name = legend_names.get(label, label)

        # 凡例の重複防止
        if legend_name not in used_labels:
            legend_label = legend_name
            used_labels.add(legend_name)
        else:
            legend_label = None

        # ===============================
        # 負方向（ペナルティ・コスト系）
        # ===============================
        if label in {"imbalance_penalty", "cost_degradation"}:
            v = -abs(val) # 確実にマイナスにする
            plt.bar(
                x_positions[i],
                v,
                bottom=bottom_neg,
                color=color,
                label=legend_label,
                zorder=3 # 0のラインより上に描画するための設定
            )
            bottom_neg += v

        # ===============================
        # 正方向（収益系）
        # ===============================
        else:
            v = abs(val) # 確実にプラスにする
            plt.bar(
                x_positions[i],
                v,
                bottom=bottom_pos,
                color=color,
                label=legend_label,
                zorder=3
            )
            bottom_pos += v

    # ===============================
    # 合計値表示（純利益）
    # ===============================
    total = bottom_pos + bottom_neg

    # 上向きのバーの頂上に純利益（合計）を表示
    # plt.text(
    #     x_positions[i],
    #     bottom_pos,
    #     f"{total:.1f}",
    #     ha="center",
    #     va="bottom",
    #     fontweight="bold",
    #     zorder=4
    # )

# ===============================
# 軸設定
# ===============================
plt.xticks(x_positions, x_labels, fontsize=14)
plt.axhline(0, color='black', linewidth=1, zorder=2) # Y=0の基準線を少し強調

plt.ylabel("Average Revenue ($)", fontsize=14)
plt.yticks(fontsize=12)
plt.ylim(-10000, 60000)  # 下方向も表示するため調整

# 凡例
plt.legend(
    bbox_to_anchor=(0.5, 1.15),
    loc="upper center",
    fontsize=12,
    ncol=3 # 全5要素になるので、見栄えを考慮して3列に
)

plt.tight_layout()

# 保存
plt.savefig(
    "/home/students1/mantani/IEEE_TEMPR_value_stacking/plot/plot_hybrid_vs_colocated.pdf"
)

plt.show()