import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# データ読み込み
# ======================
market_data = pd.read_csv(
    "./data/caiso_hourly_prices_2022.csv",
    encoding="shift_jis"
)

ancillary_data = pd.read_csv(
    "./data/caiso_as_prices_2022.csv",
    encoding="shift_jis"
)
pv_actual = pd.read_csv(
    "./data/caiso_solar_hourly_2022.csv",
    encoding="shift_jis"
)


#############################

timestamp, seed, pv_capacity, casename = ("0414_1648", 1, 14.92, "case1")
timestamp, seed, pv_capacity, casename = ("0410_0948", 1, 24.77, "case2")
timestamp, seed, pv_capacity, casename = ("0410_1619", 1, 19.41, "case3")
timestamp, seed, pv_capacity, casename = ("0414_1111", 4, 11.53, "case4") # Highest reward


datapath= "./codesign_ddpg_logs/" + f"run_{timestamp}_hybrid_PV_ratio_1.5_seed_{seed}/action/"

#################################

b_en = pd.read_csv(
    datapath + f"episode_b_en_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

b_up = pd.read_csv(    
    datapath + f"episode_b_up_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

b_dn = pd.read_csv(
    datapath + f"episode_b_dn_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

b_res = pd.read_csv(
    datapath + f"episode_b_res_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

soc = pd.read_csv(
    datapath + f"episode_soc_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

E_short = pd.read_csv(
    datapath + f"episode_E_short_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

a_imb = pd.read_csv(
    datapath + f"episode_a_imb_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

p_pred = pd.read_csv(
    datapath + f"episode_p_pred_{timestamp}_{seed}.csv",
    encoding="shift_jis"
)

# ======================
# numpy化
# ======================
plot_range = 24 * 7
time_steps = np.arange(plot_range)
pv_vals      = pv_actual.iloc[:, 1].to_numpy() / 14000.0  # GW → MW換算済想定
pv_vals = pv_vals[4344:4344+24*7]
soc_vals = soc.iloc[-1, :].to_numpy()
a_imb_vals = a_imb.iloc[-1, :].to_numpy()
p_pred_vals = p_pred.iloc[-1,:].to_numpy() 

price_energy = market_data.iloc[:, 1].to_numpy()[4344:4344+plot_range]

coef_up = 3.0 if casename=="case4" else 1.0
coef_dn = 0.5 if casename=="case4" else 1.0
price_up = ancillary_data.iloc[:, 2].to_numpy()[4344:4344+plot_range] * coef_up
price_dn = ancillary_data.iloc[:, 3].to_numpy()[4344:4344+plot_range] * coef_dn
price_sr = ancillary_data.iloc[:, 1].to_numpy()[4344:4344+plot_range] * coef_up

b_en_vals = b_en.iloc[-1, :].to_numpy()[:plot_range]
b_up_vals = b_up.iloc[-1, :].to_numpy()[:plot_range]
b_dn_vals = b_dn.iloc[-1, :].to_numpy()[:plot_range]
b_res_vals = b_res.iloc[-1, :].to_numpy()[:plot_range]


# ======================
# 正負分解
# ======================
b_en_pos = np.clip(b_en_vals, 0, None)
b_en_neg = np.clip(b_en_vals, None, 0)

# ======================
# プロット
# ======================
fig, ax1 = plt.subplots(figsize=(13,5))

# ----- 容量（左軸） -----
bottom_pos = np.zeros(plot_range)
bottom_neg = np.zeros(plot_range)

# 上方向
ax1.bar(time_steps, b_en_pos, bottom=bottom_pos, color="green",label="b_en", alpha=0.8)
bottom_pos += b_en_pos

ax1.bar(time_steps, b_up_vals, bottom=bottom_pos,color="red", label="b_up", alpha=0.8)
bottom_pos += b_up_vals

ax1.bar(time_steps, b_res_vals, bottom=bottom_pos,color = 'blue', label="b_res", alpha=0.8)
bottom_pos += b_res_vals

# 
ax1.bar(time_steps, b_en_neg, bottom=bottom_neg, color = 'green', label="", alpha=0.8)
bottom_neg += b_en_neg
ax1.bar(time_steps, -b_dn_vals, bottom=bottom_neg, color = 'orange',label="b_dn",alpha=0.8)
bottom_neg += - b_dn_vals  

# ax1.bar(time_steps, -b_dn_vals, bottom=bottom_neg,color = 'orange', label="b_dn", alpha=0.8)
# bottom_neg += -b_dn_vals
# ax1.bar(time_steps, b_en_neg, bottom=bottom_neg,color = 'green',label="",alpha=0.8)
#bottom_neg += b_en_neg 

ax1.plot(time_steps, pv_vals[:plot_range] * pv_capacity,
          label="actual PV power", linestyle="--", color="black")

# ax1.plot(time_steps, p_pred_vals[:plot_range], label="PVpred", linestyle = '--',color="green", linewidth=2)

# ax1.plot(time_steps, soc_vals[:plot_range] * 10,
#         label=r"$soc$", linestyle="-", color="red")

ax1.set_ylabel("Capacity (MW)", fontsize=14)
ax1.set_xlim(0,170)
ax1.axhline(0)
ax1.set_ylim(-30,30)
# ----- 価格（右軸） -----
ax2 = ax1.twinx()
ax2.set_ylim(-20,120)
ax2.plot(time_steps, price_energy, label="Energy price", color="green", linewidth=2)
ax2.plot(time_steps, price_sr, label="Spinning-Reserve price",color = 'blue')
ax2.plot(time_steps, price_up, label="Reg-Up price",color="red")
ax2.plot(time_steps, price_dn, label="Reg-Down price",color = 'orange')

ax2.set_ylabel("Price($/MWh) ", fontsize=14)

ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)

# ======================
# 凡例まとめ
# ======================
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# 全部まとめる
all_handles = handles1 + handles2
all_labels = labels1 + labels2

# ラベル → handle の対応辞書
label_to_handle = dict(zip(all_labels, all_handles))

# 表示したい順番を指定
order = [
    
    "b_en",
    "Energy price",
    "b_res",
    "Spinning-Reserve price",
    "b_up",
    "Reg-Up price",
    "b_dn",
    "Reg-Down price",
    "actual PV power"
]

# 順番通りに並べる
ordered_handles = [label_to_handle[l] for l in order if l in label_to_handle]

ax2.legend(ordered_handles, order,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.17),
           ncol=5,
           fontsize=12,
           frameon=False)

# ======================
# 仕上げ
# ======================
ax1.set_xlabel("Time (hours)", fontsize=14)

plt.tight_layout()

plt.savefig(
    f"./plot/plot_market_data_{casename}.pdf"
)

plt.show()

