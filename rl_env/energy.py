import numpy as np
import pandas as pd

class CaisoMarketEnv:
    def __init__(self, mode='discrete'):
        self.mode = mode
        
        # --- データ読み込み ---
        self.ancillary_data = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_as_prices_week_region_columns.csv", encoding='shift_jis')
        self.market_data = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_dam_avg_price_week.csv", encoding='shift_jis')
        self.renewable_forecast = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_renewables_forecast_solar_week_caiso.csv", encoding='shift_jis')
        self.renewable_actual   = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR_value_stacking/data/caiso_renewables_hourly_solar_week.csv", encoding='shift_jis')
        
        # --- 価格・太陽光データ ---
        self.price_energy = self.market_data.iloc[:, 1].to_numpy()
        self.price_raise  = self.ancillary_data.iloc[:, 3].to_numpy()
        self.price_lower  = self.ancillary_data.iloc[:, 4].to_numpy()
        
        self.solar_forecast_vals = self.renewable_forecast.iloc[:, 2].to_numpy()
        self.solar_actual_vals   = self.renewable_actual.iloc[:, 1].to_numpy()
        
        # --- バッテリー設定 ---
        self.capacity = max(self.solar_actual_vals)
        self.eta_c = 0.95
        self.eta_d = 0.95
        self.alpha = 1
        self.delta_t = 5/60  # 5分刻み
        
        self.E_max = self.capacity * 0.9
        self.E_min = self.capacity * 0.1
        self.E = self.capacity * 0.5
        self.t = 0
        self.T = len(self.price_energy)
        
        # --- 行動空間 ---
        self.a_energy_values = np.arange(-1.0, 1.1, 0.2)
        self.a_fcas_values   = np.arange(-1.0, 1.1, 0.2)
        self.a_solar_values  = np.arange(0.0, 1.1, 0.5)
        
        self.observation_space = 6
        self.action_space = 3 if mode=='continuous' else len(self.a_energy_values) * len(self.a_fcas_values) * len(self.a_solar_values)
        
        # --- 履歴記録 ---
        self.a_energy_history = []
        self.a_fcas_history   = []
        self.a_solar_history  = []

    def reset(self):
        self.t = 0
        self.E = self.capacity * 0.5
        self.a_energy_history = []
        self.a_fcas_history   = []
        self.a_solar_history  = []
        
        return np.array([
            self.E,
            self.price_energy[self.t],
            self.price_raise[self.t],
            self.price_lower[self.t],
            self.solar_forecast_vals[self.t],
            self.solar_actual_vals[self.t-1] if self.t>0 else 0.0
        ], dtype=np.float32)
    
    def step(self, action):
        E_t = self.E
        
        # --- 離散行動の場合 ---
        if self.mode=='discrete':
            energy_idx = action // (len(self.a_fcas_values) * len(self.a_solar_values))
            rem = action % (len(self.a_fcas_values) * len(self.a_solar_values))
            fcas_idx = rem // len(self.a_solar_values)
            solar_idx = rem % len(self.a_solar_values)
            
            a_energy = self.a_energy_values[energy_idx]
            a_fcas   = self.a_fcas_values[fcas_idx]
            a_solar  = self.a_solar_values[solar_idx]
        else:
            # --- 連続行動の場合 ---
            a_energy = np.clip(action[0], -1.0, 1.0)
            a_fcas   = np.clip(action[1], -1.0, 1.0)
            a_solar  = np.clip(action[2], 0.0, 1.0)
        
        # --- FCASの制約 ---
        max_fcas = self.E_max - E_t
        min_fcas = -E_t
        self.a_fcas = np.clip(a_fcas, min_fcas, max_fcas)
        
        # --- エネルギー制約 ---
        max_energy = self.E_max - abs(self.a_fcas) - E_t
        min_energy = -(E_t + abs(self.a_fcas))
        self.a_energy = np.clip(a_energy, min_energy, max_energy)
        
        # --- 太陽光売電・自家消費 ---
        solar_actual = self.solar_actual_vals[self.t]
        solar_sold   = solar_actual * a_solar
        solar_self   = solar_actual - solar_sold
        self.a_solar_history.append(a_solar)
        
        # --- バッテリー充放電フラグ ---
        b_c = 1 if self.a_energy > 0 else 0
        b_d = 1 if self.a_energy < 0 else 0
        
        # --- 実際のMW ---
        D = max(-self.a_energy, 0.0) * self.capacity
        C = max(self.a_energy, 0.0) * self.capacity
        R = max(-self.a_fcas, 0.0) * self.capacity
        L = max(self.a_fcas, 0.0) * self.capacity
        
        # --- FCAS信号 ---
        S = np.clip(np.random.normal(0, 0.5), -1, 1)
        
        # --- SoC更新 ---
        delta_E = b_c * (C - S*L) - b_d * (D + S*R)
        self.E = np.clip(self.E + delta_E, self.E_min, self.E_max)
        
        # --- 報酬計算 ---
        spot = self.price_energy[self.t]
        reward_energy = spot * (b_d * self.eta_d * D - b_c * C / self.eta_c)
        reward_fcas   = self.price_raise[self.t]*R + self.price_lower[self.t]*L
        reward_solar  = spot * solar_sold
        cost_deg      = self.alpha * (D + S*R)
        reward = (reward_energy + reward_fcas + reward_solar - cost_deg) * self.delta_t
        
        # --- 履歴保存 ---
        self.a_energy_history.append(self.a_energy)
        self.a_fcas_history.append(self.a_fcas)
        
        # --- 次ステップ更新 ---
        self.t += 1
        done = self.t >= self.T
        if not done:
            next_state = np.array([
                self.E,
                self.price_energy[self.t],
                self.price_raise[self.t],
                self.price_lower[self.t],
                self.solar_forecast_vals[self.t],
                self.solar_actual_vals[self.t-1] if self.t>0 else 0.0
            ], dtype=np.float32)
        else:
            next_state = np.zeros(6, dtype=np.float32)
        
        return next_state, reward, done
