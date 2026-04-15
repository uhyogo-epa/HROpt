import numpy as np
import random
from datetime import datetime
import ray
import pandas as pd
# Environment
# from rl_env.energy import RenewableEnergyEnv
from rl_env.codesign_multi_market_energy_lstm import PV_BESS_AS_codesign_lstm_Env
###############################################################################
from agent.codesign_ddpg  import  CodesignDDPGagent
###############################################################################
# ray.init(ignore_reinit_error=True)

@ray.remote
def run_training(seed, mu_Ebat, mode='co-located', codesign=True, ep=10000, H_cr=4.0, Bcost=1.0, Pres=1.0, Pup=1.0, Pdn=1.0):

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Setting
    Pbase   = 10 # 10MW base
    mu_Pbat = mu_Ebat / 4
    mu_Ppv  = 1.5
    if codesign:
        LOG_DIR = f'codesign_ddpg_logs/run_{datetime.now().strftime("%m%d_%H%M")}_{mode}_PV_ratio_{round(mu_Ppv,2)}_seed_{seed}'
    else:  
        LOG_DIR = f'comparison_logs/run_{datetime.now().strftime("%m%d_%H%M")}_{mode}_PV_ratio_{round(mu_Ppv,2)}_seed_{seed}'
    
    # Environment
    # rl_env    = RenewableEnergyEnv(mode ='continuous')
    rl_env      = PV_BESS_AS_codesign_lstm_Env(mode = mode,seed = seed, results_dir=LOG_DIR, 
                                               H_cr=H_cr, Pres=Pres, Pup=Pup, Pdn=Pdn)    
    state_dim   = rl_env.observation_space
    act_dim     = rl_env.action_space
    rho_dim     = rl_env.rho_space
    max_act     = 1
    
    #DRQN　parameter
    ACTOR_LEARN_RATE  = 1e-4
    CRITIC_LEARN_RATE = 1e-4
    DISCOUNT          = 0.99
    REPLAY_MEMORY_SIZE= 5000
    REPLAY_MEMORY_MIN = 100
    MINIBATCH_SIZE    = 32
    EXPL_NOISE        = 0.5
    TAU               = 5e-3
    POLICY_NOISE      = 0.5
    NOISE_DECAY       = 0.995
    NOISE_CLIP        = 0.5
    NOISE_MIN         = 0.01
    POLICY_FREQ       = 4
    
    agent = CodesignDDPGagent(state_dim, act_dim, rho_dim, max_act,
                 ACTOR_LEARN_RATE,
                 CRITIC_LEARN_RATE,
         		 DISCOUNT,
                 REPLAY_MEMORY_SIZE,
                 REPLAY_MEMORY_MIN,
                 MINIBATCH_SIZE,
                 EXPL_NOISE,      # Std of Gaussian exploration noise
                 TAU       ,
                 POLICY_NOISE,
                 NOISE_DECAY ,
                 NOISE_CLIP  ,
                 NOISE_MIN,
                 POLICY_FREQ )

    # LOG_DIR = None
    agent.train(rl_env,
              mode = mode,
              EPISODES      = ep,
              LOG_DIR       = LOG_DIR,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,   
              seed          = seed,
              learning_rate_mu    = 1e-5 if codesign else 0,
              learning_rate_sigma = 0,
              min_epi_codesign    = 500,
              pv_inv = Pbase,
              mu    = np.array([Pbase*mu_Ebat, Pbase*mu_Pbat, Pbase*mu_Ppv],dtype= float),
              sigma = np.array([0.2,0.2,0.2],dtype = float) if codesign else np.array([0.0000001,0.00000001,0.0000000001],dtype = float),
              cost_per_kWh_max = 241/7 * Bcost,  #$241/kWh/7yr
              cost_per_kW_max  = 372/7 * Bcost,  #$372/kWh/7yr
              cost_per_kw_pv_max = 1080/20, #$1080/kW/20yr
              scheduling_decay= 0.995)


if __name__ == "__main__":
    
     ray.init(ignore_reinit_error=True, num_cpus=10)

     #seeds = [1,2,3,4,5] 
     seeds = [1,2,3,4,5,6,7,8,9,10]  
     mu_Ebat_min = 1.5
     mu_Ebat_max = 1.5
     solar_radiation_all = pd.read_csv("./data/caiso_renewables_hourly_solar_week.csv") 
     pv_actual_vals = solar_radiation_all.iloc[:, 1].to_numpy()/ 1000
     mu_Ebat = mu_Ebat_min
     mus = [max(pv_actual_vals)]  # 直接リストで指定

     while mu_Ebat <= mu_Ebat_max:
         tasks = []
         for mu in mus:
             for seed in seeds:
 
                 # Section 5.2
                 # task = run_training.remote(seed,mu_Ebat, mode='co-located', ep=1000, codesign=False)
                 # tasks.append(task)           
                 # task = run_training.remote(seed,mu_Ebat, mode='hybrid', ep=1000, codesign=False)
                 # tasks.append(task)

                 # Section 5.3
                 task = run_training.remote(seed,mu_Ebat, mode='hybrid', codesign=True, ep=20000) # Case 1
                 #task = run_training.remote(seed,mu_Ebat, mode='hybrid', codesign=True, ep=20000, H_cr=8.0) # Case 2
                 #task = run_training.remote(seed,mu_Ebat, mode='hybrid', codesign=True, ep=20000, H_cr=8.0, Bcost=0.90) # Case 3
                 #task = run_training.remote(seed,mu_Ebat, mode='hybrid', codesign=True, ep=20000, Pres=3.0, Pup=3.0, Pdn=0.5) # Case 4
                 tasks.append(task)
            
         ray.get(tasks)
        
         mu_Ebat += 0.5  # 1000ずつ増やす
         
     ray.shutdown() #

