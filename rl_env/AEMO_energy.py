import numpy as np
import pandas as pd

class BatteryJointMarketEnv:
    def __init__(self,mode ='discrete'):
        self.mode = mode
        self.data = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/Labmeeting/3rdmeeting/data/NSW1_pre_ap_prices_20250625.csv", encoding='shift_jis')
        self.price_energy = self.data.iloc[0:, 1].to_numpy()
        self.price_raise = self.data.iloc[0:, 5].to_numpy()
        self.price_lower = self.data.iloc[0:, 9].to_numpy()
        self.capacity = 1  # バッテリー容量(MW) 蓄電池容量も最適化するなら考えるべき
        self.eta_c = 0.95    # 充電効率
        self.eta_d = 0.95    # 放電効率
        self.alpha = 1       # バッテリーの劣化コスト
        self.delta_t = 5/60  # 5分間のデータ
        self.t = 0
        self.E = self.capacity * 0.5  # 初期残量（50%）
        self.E_min = self.capacity * 0.1
        self.E_max = self.capacity * 0.9
        self.T = len(self.price_energy)
        self.observation_space = 4
        self.a_energy_values =  np.arange(-1.0, 1.1, 0.2) 
        self.a_fcas_values = np.arange(-1.0, 1.1, 0.2)
        self.action_space = 2 if mode == "continuous" else len(self.a_energy_values) * len(self.a_fcas_values)
        self.a_energy_history = []
        self.a_fcas_history = []
        
    def reset(self):
        self.t = 0
        self.E = self.capacity * 0.5
        #FCAS規制信号をランダムで発振するためにノイズを生成
        fcas_signal = np.random.normal(0, 0.5)
        self.S = np.clip(fcas_signal, -1, 1)
        
        self.a_energy_history = []
        self.a_fcas_history = []
        
        
        return np.array([
            self.E,
            self.price_energy[self.t],
            self.price_raise[self.t],
            self.price_lower[self.t],
            # self.S
        ], dtype=np.float32)
    
   
    def step(self, action):
        E_t = self.E  # 現在の残量
        if self.mode == "discrete":
            # 離散インデックスを値に変換
            a_energy_idx = action // len(self.a_fcas_values)#エネルギー市場に入札する容量(正充電,負放電)
            a_fcas_idx = action % len(self.a_fcas_values)#FCAS入札する容量(正lower,負raise)
    
            a_energy_raw = self.a_energy_values[a_energy_idx]
            a_fcas_raw = self.a_fcas_values[a_fcas_idx]
    
            # FCAS行動のclip
            max_fcas = self.E_max - E_t #満タンをこえないところまでLowerに参加できる
            min_fcas = -E_t #今の充電量をすべてRaiseに使う
            self.a_fcas = np.clip(a_fcas_raw, min_fcas, max_fcas)
            
            #エネルギー取引とFCASで準備しておく量が重ならないように制限
            # a_energyは、a_fcasに依存してclip（論文の制約式）
            
            #充電できる最大量(現在のバッテリー状態は考慮しなくていいの？)
            max_energy = self.E_max - abs(self.a_fcas) - E_t
            #放電できる最大量
            min_energy = -(E_t + abs(self.a_fcas)) 
            self.a_energy = np.clip(a_energy_raw, min_energy, max_energy)
    
        else:
            max_fcas = self.E_max - E_t
            min_fcas = -E_t
            self.a_fcas = np.clip(action[1], min_fcas, max_fcas)
            
            #充電できる最大量(現在のバッテリー状態は考慮しなくていいの？)
            max_energy = self.E_max - abs(self.a_fcas) -E_t
            #放電できる最大量
            min_energy = -(E_t + abs(self.a_fcas)) 
            self.a_energy = np.clip(action[0], min_energy, max_energy)
    
        # 行動履歴記録
        self.a_energy_history.append(self.a_energy)
        self.a_fcas_history.append(self.a_fcas)
    
        # --- b_c, b_d計算（充放電フラグ） ---
        b_c_t = 1 if self.a_energy > 0 else 0
        b_d_t = 1 if self.a_energy < 0 else 0
    
    
        # 実際のエネルギー（MW）計算
        D = max(-self.a_energy, 0.0) * self.capacity
        C = max(self.a_energy, 0.0) * self.capacity
        R = max(-self.a_fcas, 0.0) * self.capacity
        L = max(self.a_fcas, 0.0) * self.capacity
    
        # FCAS信号
        fcas_signal = np.random.normal(0, 0.5)
        S = np.clip(fcas_signal, -1, 1)
        self.S = S
    
        # SoC更新
        delta_E = b_c_t * (C - S * L) - b_d_t * (D + S * R)
        self.E = np.clip(self.E + delta_E, self.E_min, self.E_max)
    
        # 報酬計算
        spot = self.price_energy[self.t]
        reward_energy = spot * (b_d_t * self.eta_d * D - b_c_t * C / self.eta_c)
        reward_fcas = self.price_raise[self.t] * R + self.price_lower[self.t] * L
        cost_deg = self.alpha * (D + S * R)
        reward = (reward_energy + reward_fcas - cost_deg) * self.delta_t
    
        # タイムステップ更新
        self.t += 1
        done = self.t >= self.T
    
        if not done:
            next_state = np.array([
                self.E ,
                self.price_energy[self.t],
                self.price_raise[self.t],
                self.price_lower[self.t],
                # self.S,
            ], dtype=np.float32)
        else:
            next_state = np.zeros(4, dtype=np.float32)
    
        return next_state, reward, done
