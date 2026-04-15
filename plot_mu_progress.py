# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==========================================
# 1. ログディレクトリとグループの設定
# ==========================================
base_dir = "./codesign_ddpg_logs/"

log_dirs_groups = {    
    "Case 1": [os.path.join(base_dir, f"run_0414_1648_hybrid_PV_ratio_1.5_seed_{i}") for i in [1,2,3,5,6,7]],
    "Case 2": [os.path.join(base_dir, f"run_0410_0948_hybrid_PV_ratio_1.5_seed_{i}") for i in [1,2,3,4,5,6,7,8,9,10]],
    "Case 3": [os.path.join(base_dir, f"run_0410_1619_hybrid_PV_ratio_1.5_seed_{i}") for i in [1,2,4,5,6,7,8,9,10]],
    "Case 4": [os.path.join(base_dir, f"run_0414_1111_hybrid_PV_ratio_1.5_seed_{i}") for i in [2,4,7,9,10]],
}

case_costs = {
    "Case 1": {
        "cost_per_kWh": 241 / 7,
        "cost_per_kW_B": 372 / 7,
        "cost_per_kW_PV": 1080 / 20,
    },
    "Case 2": {
        "cost_per_kWh": 241 / 7,
        "cost_per_kW_B": 372 / 7,
        "cost_per_kW_PV": 1080 / 20,
    },
    "Case 3": {
        "cost_per_kWh": 241 / 7 * 0.9,   # Bcost =0.9
        "cost_per_kW_B": 372 / 7 * 0.9,
        "cost_per_kW_PV": 1080 / 20,
    },
    "Case 4": {
        "cost_per_kWh": 241 / 7,
        "cost_per_kW_B": 372 / 7,
        "cost_per_kW_PV": 1080 / 20,
    },
}



colors = {
    "Case 1": "#1f77b4",
    "Case 2": "#ff7f0e",
    "Case 3": "#2ca02c",
    "Case 4": "red",
}

linestyles = {
    "Case 1": "-",
    "Case 2": "-",
    "Case 3": "-",
    "Case 4": "-",  
}

tags_to_plot = ["mu/P_pv", "mu/P_B", "mu/E_B"]
y_labels = {"mu/E_B": "$E_\mathrm{B}$", "mu/P_B": "$P_\mathrm{B}$", "mu/P_pv": "$P_\mathrm{PV}$"}
ylim_list = {
        "mu/P_pv": (0,30),
        "mu/P_B":  (0,15),
        "mu/E_B":  (0,30),
    }

# ==========================================
# 2. データの読み込み
# ==========================================
extracted_data = {label: {tag: {} for tag in tags_to_plot} for label in log_dirs_groups.keys()}

print("--- ログの読み込みを開始します ---")
for label, log_dirs in log_dirs_groups.items():
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            print(f"Warning: Directory not found -> {log_dir}")
            continue

        event_files = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f]
        if not event_files:
            continue

        event_path = os.path.join(log_dir, event_files[0])
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()

        for tag in tags_to_plot:
            if tag in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(tag)
                for e in events:
                    if label in ["Case 1'", "Case 4'"] and e.step > 10000: 
                        continue

                    if e.step not in extracted_data[label][tag]:
                        extracted_data[label][tag][e.step] = []
                    extracted_data[label][tag][e.step].append(e.value)

print("--- データの読み込みが完了しました。描画を開始します ---")

# ==========================================
# 3. 縦に3つ並べたパネル
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(6.5, 4.5), sharex=True)

for i, (ax, tag) in enumerate(zip(axes, tags_to_plot)):
    
    ax.set_ylabel(y_labels[tag], fontsize=14)
    ax.set_xlim(0, 20000)
    ax.set_ylim(ylim_list[tag])

    for label in log_dirs_groups.keys():
        step_data = extracted_data[label][tag]

        if not step_data:
            continue

        steps = sorted(step_data.keys())
        means = [np.mean(step_data[step]) for step in steps]
        stds = [np.std(step_data[step]) for step in steps]

        ax.plot(steps, means,
        color=colors[label],
        linestyle=linestyles[label],
        linewidth=2,
        label=label)

        ax.fill_between(
            steps,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=colors[label],
            alpha=0.2
        )

    #ax.axvline(x=10000, color='gray', linestyle='--')


    # 上2つのパネルではx軸のtickとlabelを消す
    if i < 2:
        ax.tick_params(labelbottom=False)

# 一番下だけxラベルを表示
axes[-1].set_xlabel("Episode", fontsize=14)

# 凡例（上にまとめて1つ）
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 上にlegendの余白確保
plt.savefig("./plot/plot_mu_progress.pdf", bbox_inches="tight")
plt.show()


# ==========================================
# 4. 各タグの最終ステップの値を print で出力
# ==========================================
print("\n=== 最終ステップのデータ ===")
for tag in tags_to_plot:
    print(f"--- {tag} ---")
    for label in log_dirs_groups.keys():
        step_data = extracted_data[label][tag]
        if not step_data:
            print(f"  {label}: データなし")
            continue

        final_step = max(step_data.keys())
        final_values = step_data[final_step]
        mean_val = np.mean(final_values)
        formatted_seeds = [round(v, 4) for v in final_values]
        print(f"  {label:<8} (Step {final_step}): 平均 {mean_val:.4f} | 各シードの値: {formatted_seeds}")
print("============================\n")



# ==========================================
# 5. 各ケースの最終設計値からCAPEXを計算
# ==========================================
print("\n=== CAPEX（最終ステップの設計値から計算） ===")

for label in log_dirs_groups.keys():
    # 各muのデータを取得
    step_data_pv = extracted_data[label]["mu/P_pv"]
    step_data_pb = extracted_data[label]["mu/P_B"]
    step_data_eb = extracted_data[label]["mu/E_B"]

    if not step_data_pv or not step_data_pb or not step_data_eb:
        print(f"--- {label} ---")
        print("  データ不足")
        continue

    # 今のコードと同様に「Case全体の最終step」を使う
    final_step_pv = max(step_data_pv.keys())
    final_step_pb = max(step_data_pb.keys())
    final_step_eb = max(step_data_eb.keys())

    P_pv_vals = step_data_pv[final_step_pv]
    P_B_vals  = step_data_pb[final_step_pb]
    E_B_vals  = step_data_eb[final_step_eb]

    P_pv_mean = np.mean(P_pv_vals)
    P_B_mean  = np.mean(P_B_vals)
    E_B_mean  = np.mean(E_B_vals)

    # ケースごとの単価
    cost_per_kWh  = case_costs[label]["cost_per_kWh"]
    cost_per_kW_B = case_costs[label]["cost_per_kW_B"]
    cost_per_kW_PV = case_costs[label]["cost_per_kW_PV"]

    # mu が MW/MWh 単位であることを仮定
    capex_pv = P_pv_mean * 1000 * cost_per_kW_PV
    capex_pb = P_B_mean  * 1000 * cost_per_kW_B
    capex_eb = E_B_mean  * 1000 * cost_per_kWh
    capex_total = capex_pv + capex_pb + capex_eb

    print(f"--- {label} ---")
    print(f"  P_pv = {P_pv_mean:.4f} MW")
    print(f"  P_B  = {P_B_mean:.4f} MW")
    print(f"  E_B  = {E_B_mean:.4f} MWh")
    print(f"  CAPEX_PV      = ${capex_pv:.2f}")
    print(f"  CAPEX_B_power = ${capex_pb:.2f}")
    print(f"  CAPEX_B_energy= ${capex_eb:.2f}")
    print(f"  CAPEX_total   = ${capex_total:.2f}")

print("========================================\n")