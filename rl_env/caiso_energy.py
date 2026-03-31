import numpy as np
import pandas as pd

class PV_BESS_AS_Env:
    def __init__(self, mode="co-located", battery_times=1):
        self.mode = mode

        # --- データ読み込み ---
        self.market_data = pd.read_csv(
            "C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_dam_avg_price_week.csv",
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
        
        # --- 市場価格 & PVデータ ---
        self.price_energy = self.market_data.iloc[:, 1].to_numpy()  # DAM価格
        self.spinning_reserve_price = self.ancillary_data.iloc[:, 1].to_numpy()
        self.price_raise = self.ancillary_data.iloc[:, 3].to_numpy()  # Raise価格
        self.price_lower = self.ancillary_data.iloc[:, 4].to_numpy()  # Lower価格
        self.pv_actual_vals = self.pv_actual.iloc[:, 1].to_numpy() /1000 # PV実績値
        self.pv_forecast_vals = self.pv_forecast.iloc[:, 2].to_numpy() /1000# PV予測値
        
        # 蓄電池関連
        self.eta_c   = 0.95
        self.eta_d   = 0.95
        self.alpha   = 1.0  # 劣化コスト係数
        self.delta_t = 5/60
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.battery_times = battery_times
        
        # P_max (電力[kW]) と E_max (容量[kWh]) の定義を修正
        pv_peak_power = max(self.pv_actual_vals)
        self.P_max = pv_peak_power  
        self.E_max = self.P_max * self.battery_times 
        
        # 報酬・ペナルティ関連
        self.rho_pen = 1.0 # インバランス料金のペナルティ係数
        self.beta_t = 1.0 # 蓄電池利用コスト係数
        
        # Contingency FCASのディスパッチ（指令）が発生する確率
        self.p_con_dispatch = 0.01 
        
        # 出力抑制の発生確率
        self.p_curtailment = 0.1
        self.curtailed_count = 0
        self.curtailment_times = []

        if self.mode == "co-located":
            self.action_space = 3
        elif self.mode == "hybrid":
            self.action_space = 3
            
        # --- 状態・行動空間の定義 ---
        # 状態空間: [SOC, 太陽光発電量, エネルギー価格, スピニングリザーブ価格, ノンスピニングリザーブ,　FCAS Raise価格, FCAS Lower価格]
        self.observation_space = 6
        
        self.reset()
 
    def reset(self):
        """環境を初期状態にリセットする"""
        self.t = 0
        self.soc = 0.5
        self.total_reward = 0.0
        
        # 履歴データのリセット
        # Reset history lists
        self.soc_history              = []
        self.reward_history         = []
        self.curtailment_history    = []
        self.wasted_energy_history  = []
        self.pv_penalty_history     = []
        self.bess_penalty_history   = []
        
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
        current_solar_forecast   = self.pv_forecast_vals[self.t]
        current_price_energy     = self.price_energy[self.t]
        current_spinning_reserve = self.spinning_reserve_price[self.t]
        current_price_raise      = self.price_raise[self.t]
        current_price_lower      = self.price_lower[self.t]
        
        # --- バッテリー制約 ---
        self.Pcmax_t = self.E_max * (self.soc_max - self.soc) / self.delta_t
        self.Pdmax_t = self.E_max * (self.soc - self.soc_min) / self.delta_t
        
        self.Pbarc_t = max(0, min(self.P_max, self.Pcmax_t))
        self.Pbard_t = max(0, min(self.P_max, self.Pdmax_t))
        
        # Regulation FCASのAGC信号
        self.AGC_reg = np.random.uniform(-1, 1)
        
        if self.mode == 'co-located':
            self.bidding_pv    = np.clip(action[0],-self.P_max,self.P_max)
            
            self.a_fcas_reguration  = np.clip(action[1],-self.Pcmax_t,self.Pdmax_t)
            
            # --- FCAS 入札処理 ---
            R_reg = min(max(0, self.a_fcas_reguration), self.Pdmax_t)
            L_reg = min(max(0, -self.a_fcas_reguration), self.Pcmax_t)

            discharge_after_reg = max(0, self.Pdmax_t - R_reg)
            self.a_fcas_contingency = np.clip(action[2], 0, discharge_after_reg)
            R_spin_bid = self.a_fcas_contingency

            # --- FCASディスパッチのシミュレーション ---
            #contingency FCASの発生確率
            if np.random.rand() < self.p_con_dispatch:
                con_rate = 1
            else:
                con_rate = 0
                
            AGC_lower = max(0, -self.AGC_reg)
            AGC_raise = max(0, self.AGC_reg)
            
            # 【Regulation FCASによる調整量】
            # AGC信号が正の場合に放電量を計算
            regulation_discharge = max(0.0, AGC_raise) * R_reg
            # AGC信号が負の場合に充電量を計算
            regulation_charge = max(0.0, AGC_lower) * L_reg
            
            # 【Contingency FCASによる放電量】
            # これはAGC信号とは独立して、系統の緊急事態（周波数低下など）に応じて決まる
            contingency_discharge = con_rate * R_spin_bid
            
            actual_discharge_from_fcas = regulation_discharge + contingency_discharge
            
            curtailment_event = False
            forecast_diff = current_solar_gen - current_solar_forecast
            curtailment_prob = np.clip((current_solar_gen / self.P_max - 0.7) / 0.3, 0, 1)
            
            if np.random.rand() < curtailment_prob and forecast_diff > 0:
                curtailment_event   = True
                curtailment_surplus = forecast_diff * np.random.uniform(0.5, 1.0)
                self.curtailment_times.append(self.t)
            else:
                curtailment_event   = False
                curtailment_surplus = 0.0

            # --- 充放電判定 ---
            if curtailment_surplus > 0:
                # 出力抑制で余った電力を無料で充電
                available_charge_power = min(curtailment_surplus, self.Pbarc_t)
                Pc_t = available_charge_power
                Pd_t = 0.0
            else:
                # 通常運転
                Pd_t = min(max(0, self.bidding_pv + actual_discharge_from_fcas - regulation_charge), self.Pbard_t)
                Pc_t = min(max(0, -self.bidding_pv + regulation_charge - actual_discharge_from_fcas), self.Pbarc_t)

            # --- 系統との電力量 ---
            xD_t_total = Pd_t - Pc_t  # +は売電，−は買電
        
            # --- 報酬計算 ---
            if curtailment_event == True:
                self.curtailed_count += 1
                revenue_energy = 0.0
                cost_degradation = self.alpha * (Pd_t + Pc_t) 
            else:
               revenue_energy = xD_t_total * current_price_energy
               cost_degradation = self.alpha * (Pd_t + Pc_t) 
               
            revenue_fcas = (R_reg * current_price_raise) + (L_reg * current_price_lower) + (R_spin_bid * current_spinning_reserve)
            baseline_revenue = 0.0
            penalty_total = 0.0
        
            # --- 合計報酬 ---
            reward = (revenue_energy + revenue_fcas  - penalty_total - cost_degradation - baseline_revenue) * self.delta_t

        elif self.mode == 'hybrid':
            #---1.行動空間---
            self.bidding_pv = np.clip(action[0],0,max(self.pv_actual_vals))
            
            self.a_fcas_reguration = np.clip(action[1],
                            -self.Pcmax_t,
                             self.Pdmax_t)
            
            #---3.事故時の入札の自動振り分け---
            #スピニングの価格を比較してより高い方の価格を採用
            R_reg = min(max(0, self.a_fcas_reguration), self.Pdmax_t)
            L_reg = min(max(0, -self.a_fcas_reguration), self.Pcmax_t)
            
            discharge_after_reg = max(0, self.Pdmax_t - R_reg)
            self.a_fcas_contingency = np.clip(action[2], 0, discharge_after_reg)
            
            #---2.FCAS入札量をRaise/Lowerに分解
            R_con_bid = self.a_fcas_contingency
            R_spin_bid = R_con_bid

            # --- 6. FCASディスパッチのシミュレーション ---
            #contingency FCASの発生確率
            if np.random.rand() < self.p_con_dispatch:
                con_rate = 1
            else:
                con_rate = 0
                
            AGC_lower = max(0, -self.AGC_reg)
            AGC_raise = max(0, self.AGC_reg)
            
            # 【Regulation FCASによる調整量】
            # AGC信号が正の場合に放電量を計算
            regulation_discharge = max(0.0, AGC_raise) * R_reg
            # AGC信号が負の場合に充電量を計算
            regulation_charge = max(0.0, AGC_lower) * L_reg
            
            # 【Contingency FCASによる放電量】
            # これはAGC信号とは独立して、系統の緊急事態（周波数低下など）に応じて決まる
            contingency_discharge = con_rate * (R_spin_bid)
            
            actual_discharge_from_fcas = regulation_discharge + contingency_discharge
            
            curtailment_event = False
            Pc_t = min(max(0, current_solar_gen - self.bidding_pv + regulation_charge - actual_discharge_from_fcas),self.Pbarc_t)
            Pd_t = min(max(0, self.bidding_pv - current_solar_gen +  actual_discharge_from_fcas - regulation_charge),self.Pbard_t)
            
            xD_t_total = current_solar_gen - Pc_t + Pd_t
            penalty_total = self.rho_pen * abs(self.bidding_pv - xD_t_total) * current_price_energy
            revenue_energy = xD_t_total * current_price_energy - penalty_total

            # FCAS市場からの収益 (容量確保に対する支払い)
            revenue_fcas = (R_reg * current_price_raise) + (L_reg * current_price_lower) + (R_spin_bid * current_spinning_reserve) 
            
            # 蓄電池の劣化コスト
            cost_degradation = self.alpha * (Pc_t + Pd_t)
            
            baseline_revenue = 0
            
            # 合計報酬
            reward = (revenue_energy + revenue_fcas - cost_degradation - baseline_revenue) * self.delta_t
            
            # For consistent history logging
            self.curtailment_history.append(0)
            self.wasted_energy_history.append(0)
            self.pv_penalty_history.append(0) # No separate PV penalty in hybrid
            self.bess_penalty_history.append(penalty_total) # Log total penalty as BESS penalty

        # --- Common part for both modes: SOC update and return ---
        self.soc += (self.eta_c * Pc_t / self.E_max) * self.delta_t - ((1 / self.eta_d) * (Pd_t / self.E_max)) * self.delta_t
        self.soc = np.clip(self.soc, self.soc_min,self.soc_max)
        self.t += 1
        
        self.soc_history.append(self.soc)
        self.reward_history.append(reward) 
        done = False
        if self.t >= len(self.pv_actual_vals):
            done = True
            observation = np.array([self.soc, 0.0,0.0,0.0, 0.0,0.0], dtype=np.float32)  # 終了時のダミー観測値
            print(f"出力抑制発生回数: {self.curtailed_count}")
            self.curtailed_count = 0
        else:
           observation = np.array([
               self.soc,
               current_solar_gen,
               current_price_energy,
               current_spinning_reserve,
               current_price_raise,
               current_price_lower
           ], dtype=np.float32)
        return observation, reward, done
