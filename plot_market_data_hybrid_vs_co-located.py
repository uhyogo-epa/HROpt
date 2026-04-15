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

timestamp, seed, pv_capacity, casename = ("0409_1757", 6, 1.1, "hybrid")  # Fig8 (a)
#timestamp, seed, pv_capacity, casename = ("0409_1757", 8, 1.1, "co-located")  # Fig8 (b)  

datapath= "./comparison_logs/" + f"run_{timestamp}_{casename}_PV_ratio_{pv_capacity}_seed_{seed}/action/"


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

price_up = ancillary_data.iloc[:, 2].to_numpy()[4344:4344+plot_range]  
price_dn = ancillary_data.iloc[:, 3].to_numpy()[4344:4344+plot_range]
price_sr = ancillary_data.iloc[:, 1].to_numpy()[4344:4344+plot_range]  

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

ax1.bar(time_steps, -b_dn_vals, bottom=bottom_neg,color = 'orange', label="b_dn", alpha=0.8)
bottom_neg += -b_dn_vals
ax1.bar(time_steps, b_en_neg, bottom=bottom_neg,color = 'green',label="",alpha=0.8)

bottom_neg += b_en_neg 
ax1.plot(time_steps, pv_vals[:plot_range] * pv_capacity * 10,
          label="actual PV power", linestyle="--", color="black")

# ax1.plot(time_steps, p_pred_vals[:plot_range], label="PVpred", linestyle = '--',color="green", linewidth=2)

# ax1.plot(time_steps, soc_vals[:plot_range] * 10,
#         label=r"$soc$", linestyle="-", color="red")

ax1.set_ylabel("Capacity (MW)", fontsize=14)
ax1.set_xlim(0,170)
ax1.axhline(0)
ax1.set_ylim(-20,20)
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
           bbox_to_anchor=(0.5, 1.20),
           ncol=5,
           fontsize=12,
           frameon=False)

# ======================
# 仕上げ
# ======================
ax1.set_xlabel("Time (hours)", fontsize=18)

plt.tight_layout()

plt.savefig(
    f"./plot/plot_market_data_{casename}.pdf"
)

plt.show()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ======================
# # データ読み込み
# # ======================
# market_data = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/data/caiso_hourly_prices_2022.csv",
#     encoding="shift_jis"
# )

# pv_actual = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/data/caiso_solar_hourly_2022.csv",
#     encoding="shift_jis"
# )

# b_en = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_b_en_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )

# p_pred = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_p_pred_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )

# # b_e_bat = pd.read_csv(
# #    "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_b_e_bat_co-located_1_20260330_1348.csv",
# #     encoding="shift_jis"
# # )

# soc = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_soc_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )
# # ======================
# # numpy 配列化
# # ======================
# price_energy = market_data.iloc[:, 1].to_numpy()
# price_energy = price_energy[4344:4344+24*7]
# pv_vals      = pv_actual.iloc[:, 1].to_numpy() / 14000.0  # GW → MW換算済想定
# pv_vals = pv_vals[4344:4344+24*7]

# b_en_vals    = b_en.iloc[-1, :].to_numpy()
# p_pred_vals    = p_pred.iloc[-1, :].to_numpy()
# # b_e_bat_vals = b_e_bat.iloc[-1, :].to_numpy()
# soc_vals = soc.iloc[-1, :].to_numpy()
# plot_range = 24 * 7
# time_steps = np.arange(plot_range)

# # ======================
# # Plot ①
# # ======================
# fig, ax1 = plt.subplots(figsize=(13, 5))

# # ax1.set_xlabel("Time (hours)", fontsize=14)
# ax1.set_ylabel("Energy Price ($/MWh)", fontsize=18)
# ax1.plot(time_steps, price_energy[:plot_range],
#          label="Energy Price", linewidth=2, color="red")

# ax1.tick_params(axis = 'both',which = 'major', labelsize = 14)
# ax2 = ax1.twinx()
# ax2.set_ylabel("Power (MW)", fontsize=18)

# ax2.plot(time_steps, pv_vals[:plot_range] * 15,
#          label="actual PV power", linestyle="-", color="blue")

# ax2.plot(time_steps, b_en_vals[:plot_range],
#          label=r"$b^{\mathrm{en}}$", linestyle="--", color="red")

# ax2.plot(time_steps, p_pred_vals[:plot_range],
#          label=r"$p_\mathrm{{pred}}$", linestyle="--", color="blue")

# # ax2.plot(time_steps, b_e_bat_vals[:plot_range],
# #          label=r"$b^{\mathrm{e,bat}}$", linestyle="--", color="gray")

# # ax1.plot(time_steps, soc_vals[:plot_range],
# #         label=r"$soc$", linestyle="-", color="red")
# ax2.tick_params(axis = 'both',which = 'major', labelsize = 14)

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2,
#            loc="upper right", fontsize=12, bbox_to_anchor = (0.8,1.15), ncol= len(lines1 + lines2 ))
# ax1.set_xlim(0,170)
# ax1.set_ylim(0,100)
# ax2.set_ylim(-15,20)
# plt.tight_layout()
# plt.savefig(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/plot/energy_price_pv_ben_co-located_1.5.pdf"
# )
# plt.show()

# # ======================
# # データ読み込み
# # ======================
# ancillary_data = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/data/caiso_as_prices_2022.csv",
#     encoding="shift_jis"
# )

# b_up = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_b_up_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )

# b_dn = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_b_dn_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )

# b_res = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_b_res_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )
# # a_up = pd.read_csv(
# #     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_a_up_co-located_1_20260330_1348.csv",
# #     encoding="shift_jis"
# # )

# # a_dn = pd.read_csv(
# #     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_a_dn_co-located_1_20260330_1348.csv",
# #     encoding="shift_jis"
# # )



# soc = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_soc_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )
# # ======================
# # numpy 配列化
# # ======================
# price_up = ancillary_data.iloc[:, 2].to_numpy()
# price_up = price_up[4344:4344+24*7]
# price_dn = ancillary_data.iloc[:, 3].to_numpy()
# price_dn = price_dn[4344:4344+24*7]
# price_sr =  ancillary_data.iloc[:, 1].to_numpy()
# price_sr = price_sr[4344:4344+24*7]

# b_up_vals = b_up.iloc[-1, :].to_numpy()
# b_dn_vals = b_dn.iloc[-1, :].to_numpy()
# b_res_vals = b_res.iloc[-1, :].to_numpy()

# # a_up_vals = a_up.iloc[-1, :].to_numpy()
# # a_dn_vals = a_dn.iloc[-1, :].to_numpy()

# soc_vals = soc.iloc[-1, :].to_numpy()
# # ======================
# # Plot ②
# # ======================
# fig, ax1 = plt.subplots(figsize=(13, 5))

# ax1.set_xlabel("Time (hours)", fontsize=18)
# ax1.set_ylabel("AS Price ($/MW-h)", fontsize=18)
# # ax1.set_ylabel("rate", fontsize=14)

# # ti
# ax2 = ax1.twinx()
# ax2.set_ylabel("Committed Capacity (MW)", fontsize=18)

# ax2.plot(time_steps, b_up_vals[:plot_range],
#          label=r"$b^{\mathrm{up}}$", linestyle="--", color="red")

# ax2.plot(time_steps, b_dn_vals[:plot_range],
#          label=r"$b^{\mathrm{dn}}$", linestyle="--", color="blue")

# ax2.plot(time_steps, b_res_vals[:plot_range],
#          label=r"$b^{\mathrm{res}}$", linestyle="--", color="orange")


# # ax2.plot(time_steps, b_en_vals[:plot_range],
# #          label=r"$b^{\mathrm{en}}$", linestyle="--", color="orange")

# # ax1.plot(time_steps, price_up[:plot_range],
# #          label=r"$\lambda^\mathrm{up}$", linestyle="-", color="red")

# # ax1.plot(time_steps, price_dn[:plot_range],
# #          label=r"$\lambda^\mathrm{dn}$", linestyle="-", color="blue")

# # ax1.plot(time_steps, price_sr[:plot_range],
# #          label=r"$\lambda^\mathrm{res}$", linestyle="-", color="orange")

# # ax1.plot(time_steps, a_up_vals[:plot_range],
# #          label=r"$a^{\mathrm{up}}$", linestyle="-", color="red")

# # ax1.plot(time_steps, a_dn_vals[:plot_range],
# #          label=r"$a^{\mathrm{dn}}$", linestyle="-", color="blue")


# ax1.plot(time_steps, soc_vals[:plot_range] * 10,
#         label=r"$soc$", linestyle="-", color="gray")

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2,
#            loc="upper right", fontsize=12, bbox_to_anchor = (0.8,1.15), ncol= len(lines1 + lines2 ))
# ax1.set_xlim(0,170)
# ax1.set_ylim(-5,30)
# ax2.set_ylim(-5,15)
# plt.tight_layout()
# plt.savefig(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/plot/as_bup_bdn_co-located_1.5.pdf"
# )
# plt.show()

# #########################################

# a_imb = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_a_imb_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )
# E_disE = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_E_disE_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )

# E_short = pd.read_csv(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/action/episode_E_short_co-located_1_20260330_1348.csv",
#     encoding="shift_jis"
# )

# E_disE_vals = E_disE.iloc[-1, :].to_numpy()
# E_short_vals = E_short.iloc[-1, :].to_numpy()
# a_imb_vals = a_imb.iloc[-1, :].to_numpy()

# fig, ax1 = plt.subplots(figsize=(13, 5))

# ax1.set_xlabel("Time (hours)", fontsize=18)
# # ax1.set_ylabel("AS Price ($/MW-h)", fontsize=14)
# ax1.set_ylabel("rate", fontsize=18)

# # ti
# ax2 = ax1.twinx()
# ax2.set_ylabel("Committed Capacity (MW)", fontsize=18)

# ax2.plot(time_steps, E_disE_vals[:plot_range],
#          label=r"$E_{\mathrm{disE}}$", linestyle="--", color="red")

# ax2.plot(time_steps, E_short_vals[:plot_range],
#          label=r"$E_{\mathrm{short}}$", linestyle="--", color="blue")

# ax1.plot(time_steps, a_imb_vals[:plot_range],
#          label=r"$a_{\mathrm{imb}}$", linestyle="-", color="red")


# # ax2.plot(time_steps, b_en_vals[:plot_range],
# #          label=r"$b^{\mathrm{en}}$", linestyle="--", color="orange")

# # ax1.plot(time_steps, price_up[:plot_range],
# #          label=r"$price up$", linestyle="-", color="red")

# # ax1.plot(time_steps, price_dn[:plot_range],
# #          label=r"$price dn$", linestyle="-", color="blue")

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2,
#            loc="upper right", fontsize=12,bbox_to_anchor = (0.65,1.15), ncol= len(lines1 + lines2 ))

# ax1.set_xlim(0,170)
# ax1.set_ylim(-0.2,1.2)
# ax2.set_ylim(-1,6)
# plt.tight_layout()
# plt.savefig(
#     "/home/students1/mantani/IEEE_TEMPR_value_stacking/plot/imbalance_co-located_1.5.pdf"
# )
# plt.show()
