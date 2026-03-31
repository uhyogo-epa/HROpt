import numpy as np
import pandas as pd
import os
from datetime import datetime


class PV_BESS_AS_codesign_Env:
    def __init__(self, mode="co-located",seed = 1):
        self.mode = mode

        # =========================
        # Data loading
        # =========================
        data_path = "C:/Users/manta/OneDrive/ドキュメント/"
        #data_path = "/home/students2/mantani/"
        self.market_data = pd.read_csv(
            data_path+"IEEE_TEMPR_value_stacking/data/caiso_rtm_prices_week_hourly_avg.csv",
            encoding="shift_jis")
        self.ancillary_data = pd.read_csv(
            data_path+"IEEE_TEMPR_value_stacking/data/caiso_as_prices_week_region_columns.csv",
            encoding="shift_jis")
        self.pv_actual = pd.read_csv(
            data_path+"IEEE_TEMPR_value_stacking/data/caiso_renewables_hourly_solar_week.csv",
            encoding="shift_jis")
        
        self.pv_actual_norm = self.pv_actual.iloc[:,1].to_numpy() / 14000
        self.price_energy   = self.market_data.iloc[:, 1].to_numpy()
        self.price_sr       = self.ancillary_data.iloc[:, 2].to_numpy()
        self.price_up       = self.ancillary_data.iloc[:, 3].to_numpy()
        self.price_dn       = self.ancillary_data.iloc[:, 4].to_numpy()
        
        # =========================
        # BESS parameters
        # =========================
        self.eta_c   = 0.95
        self.eta_d   = 0.95
        self.delta_t = 1.0  # 1 hour
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.kappa   = 0.3

        # cost / penalty
        self.c_deg  = 1.0
        self.pi_imb = 1.0
        
        # =========================
        # AGC parameters
        # =========================
        self.N_AGC  = 900                # 4 sec signals
        self.AGC_DT = 4.0 / 3600.0       # hour

        # =========================
        # Result saving
        # =========================
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        now = datetime.now().strftime('%Y%m%d_%H%M')
        self.results_path = os.path.join(
            results_dir, f"episode_results_{mode}_{seed}_{now}.csv"
        )

        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                "revenue_energy",
                "revenue_fcas",
                "capacity_payment",
                "cost_degradation",
                "imbalance_penalty",
                "prediction_penalty"
            ]).to_csv(self.results_path, index=False)
        
        # dimention
        self.action_space      = 6
        self.rho_space         = 3
        self.observation_space = 6
        
    # =========================
    # Reset
    # =========================
    def reset(self, E_B, P_B, P_pv, P_inv):
        self.t    = 0
        self.soc  = 0.5

        self.P_pv = P_pv     # PVアレー容量
        self.P_inv = P_inv   # PV Inverter & POC 
        self.E_B  = E_B      # Battery Energy
        self.P_B  = P_B      # Battery Power

        self.total_revenue_energy    = 0.0
        self.total_revenue_fcas      = 0.0
        self.total_capacity_payment  = 0.0
        self.total_cost_degradation  = 0.0
        self.total_imbalance_penalty = 0.0
        self.total_prediction_penalty = 0.0

        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.soc,
            self.pv_actual_norm[self.t-1],
            self.price_energy[self.t-1],
            self.price_up[self.t-1],
            self.price_dn[self.t-1],
            self.price_sr[self.t-1],
        ], dtype=np.float32)

    # =========================
    # Step
    # =========================
    def step(self, action):
            # Action
            action = np.asarray(action).reshape(-1)
            a_solar, a_e, a_res, self.a_up, self.a_dn, self.a_imb = np.clip(action, 0.0, 1.0)

            # Actual Price
            lambda_ene = self.price_energy[self.t]
            lambda_up  = self.price_up[self.t]
            lambda_dn  = self.price_dn[self.t]
            lambda_sr  = self.price_sr[self.t]
            pv_actual  = self.pv_actual_norm[self.t]
            # Actual PV output
            current_solar = self.pv_actual_norm[self.t] * self.P_pv

            # =====================================
            #  AS constraints 
            # =====================================
            #
            #  PVからのAS入札可能量 ==============
            self.p_pred = a_solar * self.P_pv
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
            H_up = 0.33
            H_dn = 0.33
            H_res = 1.0

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
            agc_signal = np.random.choice([-1,-0.5,0,0.5, 1], size=self.N_AGC)
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
                       self.a_imb * (self.b_en - current_solar) * self.delta_t,
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
                    x_cur = max(b_up_pv * self.delta_t - E_chgE , 0)  
                    E_sur = (current_solar - self.b_en) * self.delta_t - E_chgE - x_cur 
                    E_imb = E_sur
                    
                    # Co-locatedの情報保存するときのためのダミーの変数
                    self.b_e_bat = 0
                    
            # インバランス & 実充放電量(Co-located)            
            else:
                # 太陽光のエネルギー市場入札量
                b_e_pv = max(self.p_pred - b_up_pv, 0)

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
                
                # インバランス量
                E_imb = (current_solar - self.p_pred) * self.delta_t
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
            revenue_energy = lambda_ene * self.b_en * self.delta_t
            
            if E_imb > 0:
                imbalance_revenue = (1 - self.pi_imb) * lambda_ene * E_imb
            else:
                imbalance_revenue = (1 + self.pi_imb) * lambda_ene * E_imb
            revenue_fcas = (lambda_sr * self.b_res + lambda_up * self.b_up + lambda_dn * self.b_dn) * self.delta_t
            
            # 容量市場
            phi_pv, H_cr, lambda_ra = 0.3, 4.0, 8.31 * 12 / 8760
            capacity_payment = lambda_ra * min(P_poi_max, phi_pv * self.P_pv + min(self.P_B, self.E_B / H_cr))
            
            degradation_cost = self.c_deg * (E_chgE + E_chgAS + self.E_disE + E_disAS)
            w_pred = 10
            prediction_penalty = w_pred * (a_solar - pv_actual)**2 * self.delta_t
            
            revenue = (revenue_energy + revenue_fcas + capacity_payment) 
            reward = (revenue_energy + revenue_fcas + capacity_payment +  imbalance_revenue - degradation_cost - prediction_penalty) / 10000
            self.total_revenue_energy += revenue_energy
            self.total_revenue_fcas += revenue_fcas
            self.total_capacity_payment += capacity_payment
            self.total_cost_degradation += degradation_cost
            self.total_imbalance_penalty += imbalance_revenue
            self.total_prediction_penalty += prediction_penalty
    
            # =========================
            # Step forward
            # =========================
            self.t += 1
            done = self.t >= len(self.pv_actual_norm)
    
            if done:
                pd.DataFrame([{
                    "revenue_energy": self.total_revenue_energy,
                    "revenue_fcas": self.total_revenue_fcas,
                    "capacity_payment": self.total_capacity_payment,
                    "cost_degradation": self.total_cost_degradation,
                    "imbalance_penalty": self.total_imbalance_penalty,
                    "prediction_penalty":self.total_prediction_penalty
                }]).to_csv(self.results_path, mode="a", header=False, index=False)
                obs = np.zeros(6, dtype=np.float32)
            else:
                obs = self._get_obs()
    
            return obs, reward, done, revenue
        
    