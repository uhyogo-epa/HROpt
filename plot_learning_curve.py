#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:26:33 2026
@author: students1
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


base_dir = "./comparison_logs/"
timestamp = "0409_1757"
top_k = 9 

# TensorBoard上のタグ名
reward_tag = "RL/Episode Reward"

# ===============================
# event file から scalar を読む
# ===============================
def load_scalar_from_event(event_file, tag):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0
        }
    )
    ea.Reload()

    tags = ea.Tags()["scalars"]
    if tag not in tags:
        raise ValueError(f"tag '{tag}' not found in {event_file}\nfound tags = {tags}")

    events = ea.Scalars(tag)

    steps = np.array([e.step for e in events], dtype=int)
    values = np.array([e.value for e in events], dtype=float)

    return steps, values


def collect_reward_series(mode_prefix, battery_times_mu_E="1.1", max_seed=10):
    series_list = []
    steps_ref = None
    seed_ids = []

    for seed in range(1, max_seed + 1):
        run_dir = os.path.join(
            base_dir,
            f"run_{timestamp}_{mode_prefix}_PV_ratio_{battery_times_mu_E}_seed_{seed}"
        )

        event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))

        if len(event_files) == 0:
            print(f"[not found] event file in: {run_dir}")
            continue

        event_file = sorted(event_files)[-1]

        try:
            steps, values = load_scalar_from_event(event_file, reward_tag)
        except Exception as e:
            print(f"[skip] seed={seed}, reason={e}")
            continue

        if len(values) == 0:
            continue

        if steps_ref is None:
            steps_ref = steps
        else:
            min_len = min(len(steps_ref), len(steps))
            steps_ref = steps_ref[:min_len]
            values = values[:min_len]
            series_list = [x[:min_len] for x in series_list]

        series_list.append(values)
        seed_ids.append(seed)

    if len(series_list) == 0:
        raise ValueError(f"{mode_prefix} の event file が1つも読めませんでした")

    data = np.vstack(series_list)  # (n_seed, n_step)

    final_rewards = data[:, -1]

    # 降順でソート
    idx = np.argsort(final_rewards)[::-1]

    # 最終 reward で上位 top_k を選択
    selected_idx = idx[:top_k]

    print(f"\n[{mode_prefix}] selected seeds (top {top_k}):")
    for i in selected_idx:
        print(f"  seed={seed_ids[i]}, final reward={final_rewards[i]:.2f}")

    data = data[selected_idx]

    return steps_ref, data


# ===============================
# データ取得
# ===============================
steps_h, data_h = collect_reward_series("hybrid")
steps_c, data_c = collect_reward_series("co-located")

# 念のため共通長にそろえる
common_len = min(len(steps_h), len(steps_c), data_h.shape[1], data_c.shape[1])
steps = steps_h[:common_len]
data_h = data_h[:, :common_len]
data_c = data_c[:, :common_len]

# 平均・分散・標準偏差
mean_h = np.mean(data_h, axis=0)
var_h  = np.var(data_h, axis=0)
std_h  = np.std(data_h, axis=0)

mean_c = np.mean(data_c, axis=0)
var_c  = np.var(data_c, axis=0)
std_c  = np.std(data_c, axis=0)

print("Hybrid seeds:", data_h.shape[0])
print("Co-located seeds:", data_c.shape[0])

# ===============================
# 図1: 平均 ± 標準偏差
# ===============================
plt.figure(figsize=(7, 4))

plt.plot(steps, mean_h, label="Hybrid (mean)", linewidth=2)
plt.fill_between(steps, mean_h - std_h, mean_h + std_h, alpha=0.25, label="")

plt.plot(steps, mean_c, label="Co-located (mean)", linewidth=2)
plt.fill_between(steps, mean_c - std_c, mean_c + std_c, alpha=0.25, label="")

plt.xlabel("Episode", fontsize=14)
plt.ylabel("Episode Reward", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0,1000)
plt.ylim(0,14)

plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("./plot/plot_learning_curve.pdf")
plt.show()

