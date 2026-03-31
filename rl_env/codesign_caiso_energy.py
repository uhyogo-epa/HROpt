import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class PV_BESS_AS_codesign_Env:
    def __init__(self, mode="co-located", battery_times=1):
        self.mode = mode

        # --- データ読み込み ---
        # 実行環境に合わせてパスを修正してください
        self.market_data = pd.read_csv(
            "C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_rtm_prices_week_hourly_avg.csv",
            encoding="shift_jis")
        self.ancillary_data = pd.read_csv(
            "C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_as_prices_week_region_columns.csv",
            encoding="shift_jis")
        self.pv_actual = pd.read_csv(
            "C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_renewables_hourly_solar_week.csv",
            encoding="shift_jis")
        self.pv_forecast = pd.read_csv(
            "C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_renewables_forecast_solar_week_caiso.csv",
            encoding="shift_jis")

        # --- 保存フォルダ設定 ---
        results_dir = "./results"
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
        self.results_path = os.path.join(results_dir, f"episode_results_{mode}_{current_datetime}.csv")

        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                "episode", "revenue_energy", "revenue_fcas", "baseline_revenue",
                "capacity_payment", "cost_degradation", "imbalance_penalty"
            ]).to_csv(self.results_path, index=False)

        # --- 市場価格 & PVデータ ---
        self.price_energy           = self.market_data.iloc[:, 1].to_numpy()
        self.spinning_reserve_price = self.ancillary_data.iloc[:, 1].to_numpy()
        self.price_raise            = self.ancillary_data.iloc[:, 3].to_numpy()
        self.price_lower            = self.ancillary_data.iloc[:, 4].to_numpy()
        self.pv_actual_vals         = self.pv_actual.iloc[:, 1].to_numpy() / 1000
        self.pv_forecast_vals       = self.pv_forecast.iloc[:, 2].to_numpy() / 1000

        # 蓄電池関連
        self.eta_c         = 0.95
        self.eta_d         = 0.95
        self.alpha         = 1.0
        self.delta_t       = 5 / 60
        self.soc_min       = 0.1
        self.soc_max       = 0.9
        self.battery_times = battery_times
        self.capacity      = max(self.pv_actual_vals)
        self.E             = self.capacity * 0.5
        self.E_min         = self.capacity * 0.1

        # 報酬・ペナルティ関連
        self.rho_pen = 1.0
        self.beta_t = 1.0
 
        # PVデータの最大値を計算（正規化用）
        self.PV_MAX = np.max(self.pv_actual_vals) if len(self.pv_actual_vals) > 0 else 1.0
        
        self.action_space = 3 
        self.observation_space = 6
        self.rho_space = 3
        self.episode_counter = 0
            
    def reset(self, rho_E, rho_P, rho_I):
        self.t         = 0
        self.soc       = 0.5
        self.EBESS_max = rho_E
        self.PBESS_max = rho_P
        self.PPV_max   = rho_I # PVインバータ容量 (co-located) または 共通インバータ容量 (hybrid)
        
        # EMA 関連
        self.tau_S = 0.9
        self.price_ema = self.price_energy[0]# ← ここが ρ_{S,0}
        
        # 履歴データ初期化
        self.E_history             = []
        self.reward_history        = []
        self.wasted_energy_history = []
        self.pv_penalty_history    = []
        self.bess_penalty_history  = []
        self.soc                   = []
        

        # 合計値リセット
        self.total_revenue_energy    = 0.0
        self.total_revenue_fcas      = 0.0
        self.total_cost_degradation  = 0.0
        self.total_capacity_payment  = 0.0
        self.total_baseline_revenue  = 0.0
        self.total_imbalance_penalty = 0.0

        observation = np.array([
            self.soc,
            self.pv_actual_vals[self.t],
            self.price_energy[self.t],
            self.price_raise[self.t],
            self.price_lower[self.t],
            self.spinning_reserve_price[self.t],
        ], dtype=np.float32)
        return observation

    def step(self, action):
        current_solar_gen        = self.pv_actual_vals[self.t]
        current_price_energy     = self.price_energy[self.t]
        current_spinning_reserve = self.spinning_reserve_price[self.t]
        current_price_raise      = self.price_raise[self.t]
        current_price_lower      = self.price_lower[self.t]

        # AGCシミュレーション
        self.AGC_reg = np.random.uniform(-1, 1)
        AGC_lower = max(0, -self.AGC_reg)
        AGC_raise = max(0, self.AGC_reg)

        if self.mode == 'co-located':
            # バッテリー制約（SOCによる制約）
            self.Pcmax_t = self.EBESS_max * (self.soc_max - self.soc) / self.delta_t
            self.Pdmax_t = self.EBESS_max * (self.soc - self.soc_min) / self.delta_t

            # BESSパワー制約（P_maxとSOC制約の小さい方）
            self.Pbarc_t = max(0, min(self.PBESS_max, self.Pcmax_t))
            self.Pbard_t = max(0, min(self.PBESS_max, self.Pdmax_t))
            
            #太陽光の予測の電力量と実際の電力量
            self.a_PV_forecast_norm  = action[0]
            a_PV_forecast            =  self.a_PV_forecast_norm * self.PPV_max
            a_pv_actual_norm         =  current_solar_gen / self.PPV_max
            
            #太陽光の発電量のエネルギー市場とFCAS市場の比率
            self.v_PV_propotion_norm = np.clip(action[1],0.0,1.0)
            
            #太陽光の発電量を用いたエネルギー市場とFCAS市場の収益
            revenue_PV = self.v_PV_propotion_norm * current_price_energy + (1 - self.v_PV_propotion_norm)
            
            #インバランス計算
            actual_output = min(a_pv_actual_norm,self.a_PV_forecast_norm)           
            imbalance_penalty = actual_output - self.rho_pen * abs(a_pv_actual_norm - self.a_PV_forecast_norm)
            
            #インバランスを含めた太陽光発電の利益
            profit_PV = revenue_PV * imbalance_penalty
            
            ############蓄電池#############
            #蓄電池を充放電どちらにするかを決定するための市場価格計算
            self.price_ema = (
                self.tau_S * self.price_ema +
                (1 - self.tau_S) * current_price_energy
            )
            
            #蓄電池の充放電決定のためのバイナリー変数
            ICh_t  = np.sign(self.price_ema - current_price_energy)
            IDch_t = np.sign(current_price_energy - self.price_ema)
            logits = [action[2], action[3]]
            p = np.softmax(logits)          
            v_ch, v_dch = p[0], p[1]
            
            a_bess_spot = np.clip(action[4], 0.0, 1.0) * self.PBESS_max
            a_bess_reg  = np.clip(action[5], 0.0, 1.0) * self.PBESS_max
            
            # === Spot arbitrage reward (Eq.31) ===
            r_bess_spot = (
                a_bess_spot
                * abs(current_price_energy - self.price_ema)
                * (
                    ICh_t  * v_ch  / self.eta_c +
                    IDch_t * v_dch * self.eta_d
                )
            )
            # === Regulation FCAS reward (Eq.32) ===
            r_bess_reg = (
                a_bess_reg *
                (
                    v_ch  / self.eta_c * current_price_lower +
                    v_dch * self.eta_d * current_price_raise
                )
            )
            
            # === Total BESS reward ===
            reward_BESS = r_bess_spot + r_bess_reg
            
            revenue = reward_BESS + revenue_PV 
            
        elif self.mode == 'hybrid':
            # --- 1. 行動空間の定義 ---
            # I_maxは共通インバータ容量として作用
            self.bidding_pv = np.clip(action[0], 0, self.I_max)
            self.a_fcas_reguration = np.clip(action[1], -self.Pcmax_t, self.Pdmax_t)
        
            # --- 2. FCAS入札の自動振り分け ---
            R_reg = min(max(0, self.a_fcas_reguration), self.Pbard_t)   
            L_reg = min(max(0, -self.a_fcas_reguration), self.Pbarc_t) 
            discharge_after_reg = max(0, self.Pbard_t - R_reg)
            self.a_fcas_contingency = np.clip(action[2], 0, discharge_after_reg)
            R_spin_bid = self.a_fcas_contingency 

            # --- 3. FCASディスパッチのシミュレーション ---
            con_rate = 1 if np.random.rand() < self.p_con_dispatch else 0
            regulation_discharge = AGC_raise * R_reg
            regulation_charge = AGC_lower * L_reg
            contingency_discharge = con_rate * R_spin_bid
            actual_discharge_from_fcas = regulation_discharge + contingency_discharge
            
            # --- 4. エネルギー市場部分（FCASを除く） ---
            Pc_energy = min(max(0, current_solar_gen - self.bidding_pv), self.Pbarc_t)
            Pd_energy = min(max(0, self.bidding_pv - current_solar_gen), self.Pbard_t)
            
            # --- 5. 充放電量の確定（排他性を保証）---
            P_bess_net_required = (Pd_energy - Pc_energy) + (actual_discharge_from_fcas - regulation_charge)

            if P_bess_net_required > 0:
                Pd_t = min(P_bess_net_required, self.Pbard_t)
                Pc_t = 0.0
            elif P_bess_net_required < 0:
                Pc_t = min(abs(P_bess_net_required), self.Pbarc_t)
                Pd_t = 0.0
            else:
                Pc_t = 0.0
                Pd_t = 0.0            
        
            # --- 6. 市場インバランス決済 ---
            lambda_sur_t = max(0.0, (1.0 - self.rho_pen) * current_price_energy) 
            lambda_def_t = (1.0 + self.rho_pen) * current_price_energy          
            
            # 1. SOが要求するトータルの電力 (dispatch)
            dispatch = self.bidding_pv + actual_discharge_from_fcas - regulation_charge
            
            # 2. PCCで計測される実績の電力 (PCC)
            PCC_raw = current_solar_gen + Pd_t - Pc_t
            
            # ⭐ 修正 3-1: PCC実績出力の共通インバータ容量によるクリップ
            # 最大AC出力は共通インバータ容量(I_max)とPCC契約容量(P_PCC_max)の小さい方
            max_ac_output = min(self.I_max, self.P_PCC_max)
            
            # PCC実績出力（放電側）を物理制約でクリップ
            PCC = min(PCC_raw, max_ac_output) if PCC_raw >= 0 else PCC_raw
            
            surplus_amount = max(0, PCC - dispatch)  
            deficit_amount = max(0, dispatch - PCC)  
            
            imbalance_settlement = (lambda_sur_t * surplus_amount) - (lambda_def_t * deficit_amount)
            imbalance_penalty = -imbalance_settlement 
            
            # --- 収益・コスト計算 ---
            revenue_energy = self.bidding_pv * current_price_energy
            revenue_fcas = ((R_reg * current_price_raise) + (L_reg * current_price_lower) + (R_spin_bid * current_spinning_reserve))
            cost_degradation = self.alpha * (Pc_t + Pd_t)
            baseline_revenue = 0.0 
            
            ra_price_per_mw_hour = 13.69
            capacity_payment = ra_price_per_mw_hour * self.P_max if self.E_max / self.P_max > 4.0 else 0.0
        
            # --- 総報酬 ---
            reward = (
                revenue_energy
                + revenue_fcas
                - cost_degradation
                + imbalance_settlement 
                + capacity_payment
            ) * self.delta_t
        
            # --- 記録用 ---
            self.wasted_energy_history.append(0)
            self.pv_penalty_history.append(0)
            self.bess_penalty_history.append(imbalance_penalty)
        
        # --- Common part for both modes: SOC update and return ---
        self.soc += (self.eta_c * Pc_t / self.E_max) * self.delta_t - ((1 / self.eta_d) * (Pd_t / self.E_max)) * self.delta_t
        self.soc = np.clip(self.soc, self.soc_min,self.soc_max)
        
        self.t += 1
        
        self.reward_history.append(reward)  
        
        done = False
        self.total_revenue_energy += revenue_energy * self.delta_t
        self.total_revenue_fcas += revenue_fcas * self.delta_t
        self.total_cost_degradation += cost_degradation * self.delta_t
        self.total_capacity_payment += capacity_payment * self.delta_t
        self.total_baseline_revenue      += baseline_revenue * self.delta_t
        self.total_imbalance_penalty += imbalance_penalty * self.delta_t
        
        if self.t >= len(self.pv_actual_vals):
            done = True
            observation = np.array([self.soc, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) 
            
            result = pd.DataFrame([{
            "episode": self.episode_counter,
            "revenue_energy": self.total_revenue_energy,
            "revenue_fcas": self.total_revenue_fcas,
            "baseline_revenue": self.total_baseline_revenue,
            "capacity_payment": self.total_capacity_payment,
            "cost_degradation": self.total_cost_degradation,
            "imbalance_penalty":self.total_imbalance_penalty
            }])
            result.to_csv(self.results_path, mode="a", header=False, index=False)
            
            self.episode_counter += 1
        else:
            t_next = self.t
            observation = np.array([
                self.soc,
                self.pv_actual_vals[t_next],
                self.price_energy[t_next],
                self.price_raise[t_next],
                self.price_lower[t_next],
                self.spinning_reserve_price[t_next]
            ], dtype=np.float32)
            
        return observation, reward, done