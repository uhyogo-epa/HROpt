# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

class PV_BESS_AS_codesign_lstm_Env:
    def __init__(self, mode="co-located",seed = 1, results_dir=None, H_cr=4.0, Pres=1.0, Pup=1.0, Pdn=1.0):
        self.mode = mode

        # =========================
        # Data loading
        # =========================
        # data_path = "C:/Users/manta/OneDrive/ドキュメント/"
        data_path = "/home/students1/mantani/"
        self.market_data = pd.read_csv(
            data_path+"IEEE_TEMPR_value_stacking/data/caiso_hourly_prices_2022.csv",
            encoding="shift_jis")
        self.ancillary_data = pd.read_csv(
            data_path+"IEEE_TEMPR_value_stacking/data/caiso_as_prices_2022.csv",
            encoding="shift_jis")
        self.pv_actual = pd.read_csv(
            data_path+"IEEE_TEMPR_value_stacking/data/caiso_solar_hourly_2022.csv",
            encoding="shift_jis")
        
        self.pv_actual_norm = self.pv_actual.iloc[:,1].to_numpy() / 14000
        self.price_energy   = self.market_data.iloc[:, 1].to_numpy()
        self.price_sr       = self.ancillary_data.iloc[:, 1].to_numpy() * Pres
        self.price_up       = self.ancillary_data.iloc[:, 2].to_numpy() * Pup
        self.price_dn       = self.ancillary_data.iloc[:, 3].to_numpy() * Pdn
        self.pv_actual_norm = self.pv_actual_norm[4344:4344+24*7]
        self.price_energy = self.price_energy[4344:4344+24*7]
        self.price_up = self.price_up[4344:4344+24*7]
        self.price_sr = self.price_sr[4344:4344+24*7]
        self.price_dn = self.price_dn[4344:4344+24*7]
        
        # =========================
        # BESS parameters
        # =========================
        self.eta_c   = 0.95
        self.eta_d   = 0.95
        self.delta_t = 1.0  # 1 hour
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.kappa   = 0.7

        # cost / penalty
        self.c_deg  = 1.0
        self.pi_imb = 1.0
                
        self.H_cr = H_cr
        
        # =========================
        # AGC parameters
        # =========================
        self.N_AGC  = 900                # 4 sec signals
        self.AGC_DT = 4.0 / 3600.0       # hour

        # =========================
        # Result saving
        # =========================
        if not results_dir:
            results_dir = "./results"
            os.makedirs(results_dir, exist_ok=True)
            now = datetime.now().strftime('%Y%m%d_%H%M')
            self.results_path = os.path.join(
                results_dir, f"episode_results_{now}_{mode}_{seed}.csv"
            )
        else: 
            os.makedirs(results_dir, exist_ok=True)
            self.results_path = os.path.join(
                results_dir, f"episode_results_{seed}.csv"
            )
            
        self.log_cnt = 0
    
        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                "revenue_energy",
                "revenue_fcas",
                "capacity_payment",
                "cost_degradation",
                "imbalance_penalty",
                "prediction_penalty"
            ]).to_csv(self.results_path, index=False)
   
        self.hidden_space = 128
        self.num_layers = 1
        self.batch_size = 1
        # dimention
        self.action_space      = 5
        self.rho_space         = 3
        self.state_space = 6
        self.observation_space = self.hidden_space*2 + self.state_space 

        self.lstm = nn.LSTM(self.state_space-1, self.hidden_space ,self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_space, self.state_space-1)        
        self.optimizer = optim.Adam( list(self.lstm.parameters()) + list(self.linear.parameters()), lr=1e-4)

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_space)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_space)
        return h0,c0
    
    
    # =========================
    # Reset
    # =========================
    def reset(self, E_B, P_B, P_pv, P_inv):
        self.t    = 0
        self.soc  = 0.5

        self.P_pv = P_pv     # PVアレー容量
        self.P_inv = P_inv   # PV Inverter & POC 
        self.E_B  = E_B if E_B>0 else 0  # Battery Energy
        self.P_B  = P_B if P_B>0 else 0  # Battery Power
        self.total_revenue_energy    = 0.0
        self.total_revenue_fcas      = 0.0
        self.total_capacity_payment  = 0.0
        self.total_cost_degradation  = 0.0
        self.total_imbalance_penalty = 0.0
        self.total_prediction_penalty = 0.0
        self.pred_price_energy_list = []
        
        self.batch_size = 1
        self.h, self.c = self.init_hidden()
        self.h_np = self.h.detach().numpy().flatten()
        self.c_np = self.c.detach().numpy().flatten()
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

    # =========================
    # Step
    # =========================
    def step(self, action):
            # =========================
            # LSTM prediction & learning
            # =========================
            
            # ---- 入力は一時刻前の実績 ----
            lstm_input = torch.tensor(
                [[
                    self.pv_actual_norm[self.t-1],
                    self.price_energy[self.t-1],
                    self.price_up[self.t-1],
                    self.price_dn[self.t-1],
                    self.price_sr[self.t-1]
                ]],
                dtype=torch.float32
            ).unsqueeze(0)  # (1,1,5)
            
            # hiddenを切る（超重要）
            self.h = self.h.detach()
            self.c = self.c.detach()
            
            # ---- 予測 ----
            lstm_out, (self.h, self.c) = self.lstm(lstm_input, (self.h, self.c))
            predicted_state = self.linear(lstm_out[:, -1, :])
            
            target_state = torch.tensor([
                self.pv_actual_norm[self.t],
                self.price_energy[self.t],
                self.price_up[self.t],
                self.price_dn[self.t],
                self.price_sr[self.t]
            ], dtype=torch.float32).unsqueeze(0)
            
            # ---- 損失 ----
            loss = F.mse_loss(predicted_state, target_state)
            # print("loss requires_grad:", loss.requires_grad)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # ---- hiddenをnumpy保存 ----
            self.h_np = self.h.detach().numpy().flatten()
            self.c_np = self.c.detach().numpy().flatten()

            # Action
            action = np.asarray(action).reshape(-1)
            a_e, a_res, self.a_up, self.a_dn, self.a_imb = np.clip(action, 0.0, 1.0)

            # Actual Price
            lambda_ene = self.price_energy[self.t]
            lambda_up  = self.price_up[self.t]
            lambda_dn  = self.price_dn[self.t]
            lambda_sr  = self.price_sr[self.t]
            pv_actual  = self.pv_actual_norm[self.t]
            
            # Actual PV output
            current_solar = pv_actual * self.P_pv

            # =====================================
            #  AS constraints 
            # =====================================
            #
            #  PVからのAS入札可能量 ==============
            pv_pred = predicted_state[0, 0].item()
            pv_pred = np.clip(pv_pred, 0.0, 1.0)
            pred_price_energy = predicted_state[0,1].item()
            self.pred_price_energy_list.append(pred_price_energy)
            self.p_pred = pv_pred * self.P_pv
            if self.mode == 'hybrid':
                M_pv = self.kappa * self.p_pred
            else:
                current_solar = min (current_solar, self.P_inv)
                M_pv = min(self.kappa * self.p_pred, self.P_inv)
            
            # BESSのエネルギー余力 
            # E_up_batt = min(self.eta_d * self.E_B * (self.soc - self.soc_min), self.P_B * self.delta_t)
            # E_dn_batt = min(self.E_B * (self.soc_max - self.soc) / self.eta_c, self.P_B * self.delta_t)
            E_up_batt = self.eta_d * self.E_B * (self.soc - self.soc_min)
            E_dn_batt = self.E_B * (self.soc_max - self.soc) / self.eta_c
            
            # --- 容量の確保の仕様 ---
            P_poi_max = self.P_inv 
            P_poi_min = - self.P_inv
            P_poi = P_poi_max - P_poi_min
            H_up = 0.25
            H_dn = 0.25
            H_res = 0.5

            # --- Contingency Reserve ---
            self.b_res = min(a_res * P_poi, self.P_B, E_up_batt / H_res)
               
            # --- Up-Reguation --- 
            M_up_batt = min(self.P_B - self.b_res,  # Power 
                            (E_up_batt - self.b_res * H_res) / H_up #Energy
                            )  # BESSの提供可能量
            
            self.b_up = min(self.a_up * (P_poi - self.b_res), M_pv + M_up_batt )
            b_up_pv = max(self.b_up - M_up_batt, 0) # PVの寄与分
            
            # --- Down-Regulation
            M_dn_batt = min(self.P_B, E_dn_batt / H_dn)    
            self.b_dn = min(self.a_dn * (P_poi - self.b_res - self.b_up), 
                            M_dn_batt + M_pv - b_up_pv)

            # =====================================
            #  Energy Market
            # =====================================            
            self.b_en = a_e * (P_poi_max - self.b_res - self.b_up) \
                        + (1-a_e) * (P_poi_min + self.b_dn)

            bdis_max = max(min(
                self.P_B - self.b_res - (self.b_up - b_up_pv),
                (E_up_batt - self.b_res * H_res - (self.b_up - b_up_pv) * H_up) / self.delta_t),
                0.0
            )

            b_dn_batt = max(self.b_dn - max(current_solar-b_up_pv,0), 0)         
            bchg_max = max(min(
                self.P_B - b_dn_batt ,
                (E_dn_batt - b_dn_batt * H_dn) / self.delta_t),
                0.0                
            )

            # 蓄電池の制約を考慮してエネルギー市場に入札
            b_en_max  = self.p_pred + bdis_max
            b_en_min  = - bchg_max                        
            self.b_en = np.clip(self.b_en, b_en_min, b_en_max)
            
            # =====================================
            #  Imbalance Settlment 
            # =====================================    

            # == Chargable Energy for Imbalance Mitigation 
            E_imb_dn = min( max(self.P_B - b_dn_batt , 0) *self.delta_t,\
                            max((E_dn_batt - b_dn_batt * H_dn), 0)
                            )

            # --- AGC実績 ---
            agc_signal = np.random.choice([-1,0,0,0,0,0,0, 1], size=self.N_AGC)
            h_up = np.sum(agc_signal[agc_signal > 0])  * self.AGC_DT
            h_dn = np.sum(agc_signal[agc_signal < 0]) * self.AGC_DT
            h_res = 1.0 if np.random.rand() < 0.5 / 168 else 0.0
                
            # インバランス & 実充放電量(Hybrid)
            self.E_disE, E_chgE,self.E_short  = 0.0, 0.0, 0.0
            if self.mode == "hybrid":   
                
                # 不足時
                if self.b_en > max(current_solar - b_up_pv, 0): 
                
                   # BESSのインバランス解消のための放電可能量
                   E_imb_up = min(
                     max(self.P_B - self.b_res - self.b_up ,0.0) * self.delta_t, 
                     max(E_up_batt - self.b_res * H_res - self.b_up * H_up, 0.0)
                   )
                   self.E_disE = min(
                       self.a_imb * (self.b_en - max(current_solar-b_up_pv,0)) * self.delta_t,
                       E_imb_up
                   )
                   
                   # 不足インバランス量
                   self.E_short = (self.b_en - max(current_solar-b_up_pv, 0))*self.delta_t -self.E_disE
                   
                   E_chgE = min(b_up_pv * self.delta_t ,E_imb_dn)
                   
                   E_imb  = - self.E_short
                   E_chgAS = min(self.b_dn * abs(h_dn), E_dn_batt - E_chgE)
                   self.b_e_bat = 0                    

                # 余剰時
                else: 
                    # 蓄電池の充電電力
                    E_chgE = min(
                        self.a_imb * (current_solar - self.b_en ) * self.delta_t,\
                        E_imb_dn)
                    
                    # 不足インバランスの計算
                    x_cur = max(b_up_pv*self.delta_t -E_chgE, current_solar*self.delta_t -E_chgE -self.P_inv, 0)  
                    
                    E_sur = (current_solar - self.b_en) * self.delta_t - E_chgE - x_cur 
                    E_imb = E_sur
                    
                    # Co-locatedの情報保存するときのためのダミーの変数
                    self.b_e_bat = 0
                    
            # インバランス & 実充放電量(Co-located)            
            else:
                # 太陽光のエネルギー市場入札量
                b_e_pv = max(min(self.p_pred,self.P_inv) - b_up_pv, 0)

                # 蓄電池のエネルギー市場入札量
                self.b_e_bat = self.b_en - b_e_pv

                if self.b_e_bat > 0:
                    self.b_e_bat = min(self.b_e_bat,  b_en_max)
                else:
                    self.b_e_bat = max(self.b_e_bat, b_en_min)
                
                # 充放電量
                self.E_disE = max(self.b_e_bat, 0.0) * self.delta_t
                E_chgE = max(-self.b_e_bat, 0.0) * self.delta_t
                
                self.b_en = self.b_e_bat + b_e_pv
                self.b_en = np.clip(self.b_en, -self.P_inv + self.b_dn, self.P_inv - self.b_res - self.b_up)
                # インバランス量
                E_imb = (current_solar - min(self.p_pred,self.P_inv)) * self.delta_t
                self.E_short = E_imb                

            # SOC更新
            E_disAS =  (self.b_up - b_up_pv)*h_up + self.b_res*h_res
            if self.mode == "hybrid":   
                E_chgAS = min(self.b_dn * abs(h_dn), E_dn_batt - E_chgE)
            else:
                E_chgAS = b_dn_batt * abs(h_dn)
            
            self.soc += (self.eta_c * (E_chgE + E_chgAS) - (self.E_disE + E_disAS) / self.eta_d) / self.E_B 
            self.soc = np.clip(self.soc, self.soc_min, self.soc_max)
            
            # --- 報酬計算 ---
            self.revenue_energy = lambda_ene * self.b_en * self.delta_t
            
            if E_imb > 0:
                self.imbalance_revenue = (1 - self.pi_imb) * lambda_ene * E_imb
            else:
                self.imbalance_revenue = (1 + self.pi_imb) * lambda_ene * E_imb
            
            self.revenue_fcas = (lambda_sr * self.b_res + lambda_up * self.b_up + lambda_dn * self.b_dn) * self.delta_t
            
            # 容量市場
            phi_pv, H_cr, lambda_ra = 0.4, self.H_cr, 8310*12/8760
            self.capacity_payment = lambda_ra * min(P_poi_max, phi_pv * self.P_pv + min(self.P_B, self.E_B / H_cr)) * self.delta_t
            
            self.degradation_cost = self.c_deg * (E_chgE + E_chgAS + self.E_disE + E_disAS)

            revenue = (self.revenue_energy + self.revenue_fcas + self.capacity_payment + self.imbalance_revenue) 
            reward = (self.revenue_energy + self.revenue_fcas + self.capacity_payment +  self.imbalance_revenue - self.degradation_cost) / 10000
            
            self.total_imbalance_penalty += self.pi_imb * lambda_ene * abs(E_imb)
            self.total_revenue_energy    += self.revenue_energy + self.imbalance_revenue + self.pi_imb*lambda_ene*abs(E_imb)
            self.total_revenue_fcas      += self.revenue_fcas
            self.total_capacity_payment  += self.capacity_payment
            self.total_cost_degradation  += self.degradation_cost
            
            # self.total_prediction_penalty += prediction_penalty
                                    
            self.t += 1
            done = self.t >= len(self.pv_actual_norm)
    
            if done:
                self.log_cnt += 1
                if self.log_cnt % 100 == 0:                
                    pd.DataFrame([{
                        "revenue_energy": self.total_revenue_energy,
                        "revenue_fcas": self.total_revenue_fcas,
                        "capacity_payment": self.total_capacity_payment,
                        "cost_degradation": self.total_cost_degradation,
                        "imbalance_penalty": self.total_imbalance_penalty,
                        "prediction_penalty":self.total_prediction_penalty
                    }]).to_csv(self.results_path, mode="a", header=False, index=False)

                obs = np.zeros(self.observation_space, dtype=np.float32)
            else:
                obs = self._get_obs()
    
            return obs, reward, done, revenue
        
    