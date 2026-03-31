import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

class MultiyearEnv:
    def __init__(self, agent_idx, mode):
        # データの読み込み
        data_path = "/home/students/mantani/"
        self.market_data = pd.read_csv(
            data_path + "IEEE_TEMPR_value_stacking/data/caiso_hourly_prices_2022.csv",
            encoding="shift_jis")
        self.ancillary_data = pd.read_csv(
            data_path + "IEEE_TEMPR_value_stacking/data/caiso_as_prices_2022.csv",
            encoding="shift_jis")
        self.pv_actual = pd.read_csv(
            data_path + "IEEE_TEMPR_value_stacking/data/caiso_solar_hourly_2022.csv",
            encoding="shift_jis")
        
        self.episode_count = 0
        self.lstm_train_episode = 10000
        self.lstm_frozen = False
        # データを1か月ごとに分割
        self.days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.hours_per_day = 24
        
        self.agent_idx = agent_idx
        self.mode = mode
        self.month_index = self.calculate_month_index(self.days_per_month, self.hours_per_day)
        
        if self.agent_idx == 'all':
            start_idx, end_idx = [0,8760]
        else:
            start_idx, end_idx = self.month_index[self.agent_idx]
            
        # ### 修正点1：価格データの正規化 ###
        self.price_norm_factor_1 = 500  # データの最大付近に設定
        self.price_norm_factor_2 = 600 # データの最大付近に設定
        self.price_norm_factor_3 = 50     
        self.pv_actual_norm_all = self.pv_actual.iloc[:, 1].to_numpy() / 14000
        self.pv_actual_norm = self.pv_actual_norm_all[start_idx:end_idx]
        
        self.price_energy_all = self.market_data.iloc[:,1].to_numpy() / self.price_norm_factor_1
        self.price_energy = self.price_energy_all[start_idx:end_idx]
        self.price_sr_all = self.ancillary_data.iloc[:, 1].to_numpy() / self.price_norm_factor_2
        self.price_sr = self.price_sr_all[start_idx:end_idx]
        self.price_up_all = self.ancillary_data.iloc[:, 2].to_numpy() / self.price_norm_factor_2
        self.price_up = self.price_up_all[start_idx:end_idx]
        self.price_dn_all = self.ancillary_data.iloc[:, 3].to_numpy() / self.price_norm_factor_3
        self.price_dn = self.price_dn_all[start_idx:end_idx]

        # BESS parameters
        self.eta_c, self.eta_d, self.delta_t = 0.95, 0.95, 1.0
        self.soc_min, self.soc_max, self.kappa = 0.1, 0.9, 0.7
        self.c_deg, self.pi_imb = 1.0, 0.5
        self.N_AGC = 900
        self.AGC_DT = 4.0 / 3600.0

        # Result saving
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        now = datetime.now().strftime('%Y%m%d_%H%M')
        self.results_path = os.path.join(results_dir, f"episode_results_{mode}_{now}.csv")

        # NN setup
        self.hidden_space = 128
        self.num_layers = 1
        self.batch_size = 1
        self.action_space, self.rho_space, self.state_space = 5, 3, 6
        self.observation_space = self.hidden_space * 2 + self.state_space 

        self.lstm = nn.LSTM(self.state_space - 1, self.hidden_space, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_space, self.state_space - 1)
        self.optimizer = optim.Adam(list(self.lstm.parameters()) + list(self.linear.parameters()), lr=1e-4)
        
    def calculate_month_index(self, days_per_month, hours_per_day):
        indices = []; start_idx = 0
        for days in days_per_month:
            end_idx = start_idx + days * hours_per_day
            indices.append((start_idx, end_idx)); start_idx = end_idx
        return indices
    
    def init_hidden(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_space)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_space)
        return h0, c0
    
    def reset(self, E_B, P_B, P_pv, P_inv):
        self.episode_count += 1
        self.t, self.soc = 0, 0.5
        self.P_pv, self.P_inv, self.E_B, self.P_B = P_pv, P_inv, E_B, P_B
        self.total_revenue_energy = self.total_revenue_fcas = self.total_capacity_payment = 0.0
        self.total_cost_degradation = self.total_imbalance_penalty = self.total_prediction_penalty = 0.0
        self.h, self.c = self.init_hidden()
        if self.agent_idx == 'all':
            self.pred_price_energy_list = []

        self.h_np, self.c_np = self.h.detach().numpy().flatten(), self.c.detach().numpy().flatten()
        return self._get_obs()
    
    def _get_obs(self):
        base_obs = np.array([
            self.soc,
            self.pv_actual_norm[self.t-1],
            self.price_energy[self.t-1],
            self.price_up[self.t-1],
            self.price_dn[self.t-1],
            self.price_sr[self.t-1],
        ], dtype=np.float32)
        
        obs = np.concatenate([
                    base_obs,
                    self.h_np.astype(np.float32),
                    self.c_np.astype(np.float32)
                ])
        
        return obs

    def step(self, action):
        # ### 修正点：ActionのNaNチェックとクリップ ###
        action=np.clip(np.asarray(action).reshape(-1),0.0,1.0)

        a_e,a_res,self.a_up,self.a_dn,self.a_imb=action
        # if self.agent_idx == 'all':
        #     print(f"t={self.t}, a_e={a_e:.3f}, a_dn={self.a_dn:.3f}")
        lstm_input = torch.tensor(
            [[[
                self.pv_actual_norm[self.t-1],
                self.price_energy[self.t-1],
                self.price_up[self.t-1],
                self.price_dn[self.t-1],
                self.price_sr[self.t-1]
            ]]],
            dtype=torch.float32
        )
        self.h, self.c = self.h.detach(), self.c.detach()
        lstm_out, (self.h, self.c) = self.lstm(lstm_input, (self.h, self.c))
        predicted_state = self.linear(lstm_out[:, -1, :])
        
        target_state = torch.tensor([[
            self.pv_actual_norm[self.t], self.price_energy[self.t],
            self.price_up[self.t], self.price_dn[self.t], self.price_sr[self.t]
        ]], dtype=torch.float32)
        
        # ### 修正点：Huber Loss と 勾配クリッピング ###
        loss = F.mse_loss(predicted_state, target_state)
        if self.episode_count <= self.lstm_train_episode:

           self.optimizer.zero_grad()
           loss.backward()

           torch.nn.utils.clip_grad_norm_(
           list(self.lstm.parameters()) + list(self.linear.parameters()),
             1.0
           )

           self.optimizer.step()
        elif not self.lstm_frozen:

             for p in self.lstm.parameters():
                 p.requires_grad=False

             for p in self.linear.parameters():
                 p.requires_grad=False
   
             self.lstm_frozen=True
        self.h_np, self.c_np = self.h.detach().numpy().flatten(), self.c.detach().numpy().flatten()

        # 報酬計算用に価格をデコード（元のスケールに戻す）
        lambda_ene = self.price_energy[self.t] * self.price_norm_factor_1
        lambda_up  = self.price_up[self.t] * self.price_norm_factor_2
        lambda_dn  = self.price_dn[self.t] * self.price_norm_factor_3
        lambda_sr  = self.price_sr[self.t] * self.price_norm_factor_2
        pv_actual  = self.pv_actual_norm[self.t]
        current_solar = pv_actual * self.P_pv

        # ### 修正点：予測値のNaNガードとクリップ ###
        pv_pred = predicted_state[0, 0].item()
        pv_pred = np.clip(pv_pred, 0.0, 1.0)
        pred_price_energy = predicted_state[0,1].item()
        if self.agent_idx == 'all':
           self.pred_price_energy_list.append(pred_price_energy)
        self.p_pred = pv_pred * self.P_pv

        # =====================================
        # AS / Energy Market 物理制約計算 (元のロジックを維持)
        # =====================================
        if self.mode == 'hybrid':
            M_pv = self.kappa * self.p_pred
        else:
            current_solar = min(current_solar, self.P_inv)
            M_pv = min(self.kappa * self.p_pred, self.P_inv)
        
        E_up_batt = self.eta_d * self.E_B * max(0, self.soc - self.soc_min)
        E_dn_batt = self.E_B * max(0, self.soc_max - self.soc) / self.eta_c
        
        P_poi_max, P_poi_min = self.P_inv, -self.P_inv
        P_poi = 2 * P_poi_max 
        H_up, H_dn, H_res = 0.25, 0.25, 0.5

        self.b_res = min(a_res * P_poi, self.P_B, E_up_batt / H_res)
        M_up_batt = min(self.P_B - self.b_res, (E_up_batt - self.b_res * H_res) / H_up)
        self.b_up = min(self.a_up * (P_poi - self.b_res), M_pv +  M_up_batt)
        b_up_pv = max(self.b_up -  M_up_batt, 0)
        
        M_dn_batt = min(self.P_B, E_dn_batt / H_dn)
        self.b_dn = min(self.a_dn * (P_poi - self.b_res - self.b_up), M_dn_batt + M_pv - b_up_pv)

        # エネルギー市場
        self.b_en = a_e * (P_poi_max - self.b_res - self.b_up) + (1-a_e) * (P_poi_min + self.b_dn)
        bdis_max = max(min(self.P_B - self.b_res - (self.b_up - b_up_pv), (E_up_batt - self.b_res * H_res - (self.b_up - b_up_pv) * H_up) / self.delta_t), 0.0)
        b_dn_batt = max(self.b_dn - max(current_solar-b_up_pv, 0), 0)
        bchg_max = max(min(self.P_B - b_dn_batt, (E_dn_batt - b_dn_batt * H_dn) / self.delta_t), 0.0)
        self.b_en = np.clip(self.b_en, -bchg_max, self.p_pred + bdis_max)
        
        # インバランス計算
        E_imb_dn = min(max(self.P_B - b_dn_batt, 0) * self.delta_t, max(E_dn_batt - b_dn_batt * H_dn, 0))
        agc_signal = np.random.choice([-1,0,0,0,0,0,0, 1], size=self.N_AGC)
        h_up = np.sum(agc_signal[agc_signal > 0]) * self.AGC_DT
        h_dn = np.sum(agc_signal[agc_signal < 0]) * self.AGC_DT
        h_res = 1.0 if np.random.rand() < 0.5 / 168 else 0.0

        # インバランス & 実充放電
        self.E_disE, E_chgE, self.E_short = 0.0, 0.0, 0.0
        if self.mode == "hybrid":
            if self.b_en > max(current_solar - b_up_pv, 0):
                E_imb_up = min(max(self.P_B - self.b_res - self.b_up, 0.0) * self.delta_t, max(E_up_batt - self.b_res * H_res - self.b_up * H_up, 0.0))
                self.E_disE = min(self.a_imb * (self.b_en - current_solar) * self.delta_t, E_imb_up)
                self.E_short = (self.b_en - max(current_solar-b_up_pv, 0))*self.delta_t - self.E_disE
                E_chgE = min(b_up_pv * self.delta_t, E_imb_dn)
                E_imb = - self.E_short
            else:
                E_chgE = min(self.a_imb * (current_solar - self.b_en) * self.delta_t, E_imb_dn)
                E_imb = (current_solar - self.b_en) * self.delta_t - E_chgE - max(b_up_pv * self.delta_t - E_chgE, 0)
        else:
            b_e_pv = max(self.p_pred - b_up_pv, 0)
            self.b_e_bat = np.clip(self.b_en - b_e_pv, -bchg_max, bdis_max)
            self.E_disE, E_chgE = max(self.b_e_bat, 0.0) * self.delta_t, max(-self.b_e_bat, 0.0) * self.delta_t
            self.b_en = self.b_e_bat + b_e_pv
            E_imb = (current_solar - self.p_pred) * self.delta_t

        # SOC更新
        E_disAS = (self.b_up - b_up_pv)*h_up + self.b_res*h_res
        E_chgAS = min(self.b_dn * abs(h_dn), E_dn_batt - E_chgE) if self.mode == "hybrid" else b_dn_batt * abs(h_dn)
        
        
        self.soc += (self.eta_c * (E_chgE + E_chgAS) - (self.E_disE + E_disAS) / self.eta_d) / self.E_B 
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)
        
        # 報酬計算
        self.revenue_energy = lambda_ene * self.b_en * self.delta_t
        if E_imb > 0:
            self.imbalance_revenue = (1 - self.pi_imb) * lambda_ene * E_imb
        else:
            self.imbalance_revenue = (1 + self.pi_imb) * lambda_ene * E_imb
        self.revenue_fcas = (lambda_sr * self.b_res + lambda_up * self.b_up + lambda_dn * self.b_dn) * self.delta_t
        
        phi_pv, H_cr, lambda_ra = 0.4, 4.0, 11.54
        self.capacity_payment = lambda_ra * min(P_poi_max, phi_pv * self.P_pv + min(self.P_B, self.E_B / H_cr))
        self.degradation_cost = self.c_deg * (E_chgE + E_chgAS + self.E_disE + E_disAS)

        # ### 修正点：報酬のNaNガード ###
        revenue = self.revenue_energy + self.revenue_fcas + self.capacity_payment
        reward_sum = (self.revenue_energy + self.revenue_fcas + self.capacity_payment + self.imbalance_revenue - self.degradation_cost)
        if self.agent_idx == 'all':
            reward = reward_sum /10000
        else:
            reward = reward_sum * 365 / (self.days_per_month[self.agent_idx] * 10000)
            
        # 保存
        self.total_revenue_energy += self.revenue_energy
        self.total_revenue_fcas += self.revenue_fcas
        self.t += 1
        if self.agent_idx == 'all' and self.t % 1000 ==0:
           print(
           "a_e", a_e,
           "b_res", self.b_res,
           "b_up", self.b_up,
           "b_dn", self.b_dn,
           "b_en_raw", a_e * (P_poi_max - self.b_res - self.b_up) + (1-a_e) * (P_poi_min + self.b_dn),
           "bchg_max", bchg_max,
           "bdis_max", bdis_max,
           "soc", self.soc
       )
        done = self.t >= len(self.pv_actual_norm) 
        if done and self.agent_idx == 'all':
            df = pd.DataFrame({
                "t": np.arange(len(self.pred_price_energy_list)),
                "pred_price_energy": self.pred_price_energy_list
            })
            
            df.to_csv(self.results_path.replace(".csv","_pred_price_energy.csv"), index=False)
        obs = self._get_obs() if not done else np.zeros(self.observation_space, dtype=np.float32)
    
        return obs, reward, done, revenue
