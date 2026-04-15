import numpy as np
import pandas as pd
import os
from datetime import datetime


class PV_BESS_AS_codesign_Env_old:
    def __init__(self, mode="co-located",seed = 1):
        self.mode = mode

        # =========================
        # Data loading
        # =========================
        self.market_data = pd.read_csv(
            "/home/students2/mantani/IEEE_TEMPR_value_stacking/data/caiso_rtm_prices_week_hourly_avg.csv",
            encoding="shift_jis")
        self.ancillary_data = pd.read_csv(
            "/home/students2/mantani/IEEE_TEMPR_value_stacking/data/caiso_as_prices_week_region_columns.csv",
            encoding="shift_jis")
        self.pv_actual = pd.read_csv(
            "/home/students2/mantani/IEEE_TEMPR_value_stacking/data/caiso_renewables_hourly_solar_week.csv",
            encoding="shift_jis")
        
        #規格化
        self.pv_actual_norm = self.pv_actual.iloc[:, 1].to_numpy() / 14000
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
        self.kappa   = 0.5

        # cost / penalty
        self.c_deg  = 1.0
        self.pi_imb = 3.0
        
        # =========================
        # AGC parameters
        # =========================
        self.N_AGC  = 900                # 4 sec signals
        self.AGC_DT = 4.0 / 3600.0       # hour

        # =========================
        # Result saving０
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
        
        #設計変数
        self.P_pv = P_pv
        self.P_inv = P_inv
        self.E_B  = E_B
        self.P_B  = P_B

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
            self.pv_actual_norm[self.t],
            self.price_energy[self.t],
            self.price_up[self.t],
            self.price_dn[self.t],
            self.price_sr[self.t],
        ], dtype=np.float32)

    # =========================
    # Step
    # =========================
    def step(self, action):
            action = np.asarray(action).reshape(-1)
            a_solar, a_e, a_res, self.a_up, self.a_dn, self.a_imb = np.clip(action, 0.0, 1.0)
            self.a_imb = 1.0
            lambda_ene = self.price_energy[self.t]
            lambda_up  = self.price_up[self.t]
            lambda_dn  = self.price_dn[self.t]
            lambda_sr  = self.price_sr[self.t]
            
            current_solar_norm = self.pv_actual_norm[self.t]
            current_solar = current_solar_norm * self.P_pv
            
            pv_prediction_norm = a_solar
            pv_prediction = pv_prediction_norm * self.P_pv
            
            self.p_pred = pv_prediction

            # --- 実出力 ---
            P_poi_max = self.P_inv 
            P_poi_min = - self.P_inv
            
            P_poi = P_poi_max - P_poi_min
            
            #a_solarにより入札に使用するPV量を決定
            if self.mode == 'hybrid':
                M_pv = self.kappa * self.p_pred
            else:
                M_pv = min(self.kappa * self.p_pred, P_poi_max)
               
            # --- AGC実績 ---
            agc_signal = np.random.choice([-1,-0.5,0,0.5, 1], size=self.N_AGC)
            h_up = np.sum(agc_signal[agc_signal > 0])  * self.AGC_DT
            h_dn = np.sum(agc_signal[agc_signal < 0]) * self.AGC_DT
            h_res = 1.0 if np.random.rand() < 0.5 / 168 else 0.0
            H_up = 0.33
            H_dn = 0.33
            H_res = 1.0
            
            # --- BESS供給能力 ---
            # E_up_batt = min(self.eta_d * self.E_B * (self.soc - self.soc_min), self.P_B * self.delta_t)
            # E_dn_batt = min(self.E_B * (self.soc_max - self.soc) / self.eta_c, self.P_B * self.delta_t)
           
            E_up_batt = self.eta_d * self.E_B * (self.soc - self.soc_min)
            E_dn_batt = self.E_B * (self.soc_max - self.soc) / self.eta_c
                       
            # --- 容量割当 (Algorithm 1) ---

            self.b_res = min(a_res * P_poi, self.P_B, E_up_batt / H_res)
               
            #self.b_up = min(self.a_up * (self.P_B - self.b_res), (E_up_batt + M_pv - self.b_res * H_res) / h_up )
            
            #---BESS最大提供上げ下げ容量
            M_up_batt = min(self.P_B - self.b_res, (E_up_batt - self.b_res * H_res) / H_up)
            M_dn_batt = min(self.P_B, E_dn_batt / H_dn)
            self.b_up = min(self.a_up * (P_poi - self.b_res), M_pv + M_up_batt )
            
            b_up_pv = max(self.b_up - M_up_batt, 0)
           
            
           # PVがAS(Up)にどれだけ使われるか（実績）
            # deltaE_up = max(self.b_res * H_res + self.b_up * h_up - E_up_batt, 0.0)
            # M_pv_for_energy = self.p_pred * self.delta_t - deltaE_up
            
    
            #self.b_dn = min(self.a_dn * (self.P_B - self.b_res - self.b_up), (E_dn_batt + M_pv_for_energy) / h_dn )
            #self.b_dn = min(min(self.a_dn * (P_poi - self.b_res - self.b_up), self.P_B - self.b_res -self.b_up),(E_dn_batt + M_pv_for_energy) / h_dn )
            self.b_dn = min(self.a_dn * (P_poi - self.b_res - self.b_up), M_dn_batt + M_pv - b_up_pv)
            
            # --- 入札と実出力 ---
            self.b_en = a_e * (P_poi_max - self.b_res - self.b_up) + (1-a_e) * (P_poi_min + self.b_dn)
            
            b_dn_batt = max(self.b_dn - (current_solar - b_up_pv), 0)
          
            E_solar = current_solar * self.delta_t
            
            
            # インバランス補正
            self.E_disE, E_chgE,self.E_short  = 0.0, 0.0, 0.0
            if self.mode == "hybrid":   
                self.b_en = min(self.b_en, self.p_pred + E_up_batt / self.delta_t)
                
                if self.b_en > current_solar - b_up_pv: # 不足時
                
                   E_imb_up = min(
                   max(self.P_B - self.b_res - self.b_up ,0.0) * self.delta_t, 
                   max(E_up_batt - self.b_res * H_res - self.b_up * H_up, 0.0)
                   )
                   
                   self.E_disE = min(
                       self.a_imb * (self.b_en * self.delta_t - E_solar),
                       E_imb_up
                   )
                   
                   self.E_short = self.b_en * self.delta_t - E_solar + b_up_pv * self.delta_t - self.E_disE
                   
                   E_imb_dn = min(max(self.P_B - b_dn_batt , 0 ) *self.delta_t, max((E_dn_batt - b_dn_batt * H_dn),0))
                   
                   E_chgE = min(b_up_pv * self.delta_t ,E_imb_dn)
                   
                   E_imb  = - self.E_short
                   E_chgAS = min(self.b_dn * abs(h_dn), E_dn_batt - E_chgE)
                   self.b_e_bat = 0
                    #     E_up_batt - (self.b_res * H_res + self.b_up * h_up),
                    #     0.0
                    # )
                    
                    # self.E_disE = min(
                    #     self.a_imb * (self.b_en * self.delta_t - E_solar),
                    #     E_dis_bar
                    # )
                    
                    # # self.E_short = E_sched - E_solar - self.E_disE + deltaE_up
                    
                    # DeltaE_B = max(
                    #     self.b_dn * h_dn - (E_solar - deltaE_up),
                    #     0.0
                    # )
                    
                    # E_chgE = min(deltaE_up, E_dn_batt - DeltaE_B)
                    
                    # self.E_short = (self.b_en - p_solar) * self.delta_t - self.E_disE + deltaE_up
                    

                else: # 余剰時
                    # Eq.(35): Down regulation に必要なエネルギー
                    E_imb_dn = min(max(self.P_B - b_dn_batt , 0 ) *self.delta_t, max((E_dn_batt - b_dn_batt * H_dn),0))
                    
                    
                    # Eq.(37): 余剰エネルギーの充電
                    E_chgE = min(
                        self.a_imb * (E_solar - self.b_en * self.delta_t),
                        E_imb_dn
                    )
                    
                    x_cur = max(b_up_pv * self.delta_t - E_chgE,0)
                    E_sur = E_solar - self.b_en * self.delta_t - E_chgE - x_cur 
                    
                    E_imb = E_sur
                    E_chgAS = min(self.b_dn * abs(h_dn), E_dn_batt - E_chgE)
                    self.b_e_bat = 0
            
            else:    
                # (subtract PV used for ancillary services)
                # # --- PV energy bid (Eq.54)
                #b_e_pv = self.p_pred - b_up_pv
                b_e_pv = self.p_pred - b_up_pv
                # # --- Residual battery bid
               
                # b_en_max = self.p_pred - b_up_pv
                
                # self.b_en = min(self.b_en,b_en_max)
                
                # --- Battery feasible limits (Eq.57)
                bdis_max = min(
                    self.P_B - self.b_res - (self.b_up - b_up_pv),
                    (E_up_batt - self.b_res * H_res - (self.b_up - b_up_pv) * H_up) / self.delta_t
                )
            
                bchg_max = min(
                    self.P_B - b_dn_batt ,
                    (E_dn_batt - b_dn_batt * H_dn) / self.delta_t
                )
                # b_en_max = b_e_pv + max(bdis_max, 0.0)
    
                # # エネルギー市場入札量を制限
                # self.b_en = np.clip(self.b_en, 0.0, b_en_max)
                self.b_e_bat = self.b_en - b_e_pv
                
                # --- Clip battery energy bid (Eq.56)
                if self.b_e_bat > 0:
                    self.b_e_bat = min(self.b_e_bat, bdis_max)
                else:
                    self.b_e_bat = max(self.b_e_bat, -bchg_max)
                
                # --- Battery energy for SOC update (no imbalance)
                self.E_disE = max(self.b_e_bat, 0.0) * self.delta_t
                E_chgE = max(-self.b_e_bat, 0.0) * self.delta_t
                
                self.b_en = self.b_e_bat + b_e_pv
                
                # --- PV imbalance only (Eq.55)
                E_imb = (current_solar - self.p_pred) * self.delta_t
                self.E_short = E_imb
                E_chgAS = b_dn_batt * abs(h_dn)

            # SOC更新
            p_up    = self.b_up - b_up_pv
            E_disAS =  p_up * h_up + self.b_res * h_res
            
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
            
            w_pred = 2000.0
            prediction_penalty = w_pred * (pv_prediction_norm - current_solar_norm)**2 * self.delta_t
            
            revenue = (revenue_energy + revenue_fcas + capacity_payment) 
            reward = (revenue_energy + revenue_fcas + capacity_payment + imbalance_revenue - degradation_cost - prediction_penalty) / 10000
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
        
    