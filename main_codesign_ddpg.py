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
def run_training(seed, battery_times_mu_pv):
# if __name__ == "__main__":
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    # start_time = datetime.now()
    
    # log_dir = 'logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    # writer  = SummaryWriter(log_dir=log_dir)
    
    # Environment
    # rl_env      = RenewableEnergyEnv(mode ='continuous')
    mode = 'co-located'
    rl_env      = PV_BESS_AS_codesign_lstm_Env(mode = mode,seed = seed)
    
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
    
    # ###############################################
    # # Codesign learning parameters
    # ###############################################
    # learning_rate_mu    = 1e-6
    # learning_rate_sigma = 0
    # min_epi_codesign    = 300
    # rho_list = []
    # reward_list = []
    # mu    = 0.1
    # sigma = 0.2
    # battery_price = 1000

    
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
    
    battery_times_mu_E = 2.0
    battery_times_mu_P = 0.5
    # battery_times_mu_pv = 1.0
    mu = 10
    LOG_DIR = f'codesign_ddpg_logs_{datetime.now().strftime("%m%d")}_{round(battery_times_mu_pv,2)}/test_run_{datetime.now().strftime("%m%d_%H%M")}_seed_{seed}_battery_times_{round(battery_times_mu_pv,2)}_{mode}'
    # LOG_DIR = None
    agent.train(rl_env,
              mode = mode,
              EPISODES      = 4000,
              LOG_DIR       = LOG_DIR,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,   
              seed          = seed,
              learning_rate_mu    = 1e-5,
              learning_rate_sigma = 0,
              min_epi_codesign    = 300,
              pv_inv = mu,
              mu    = np.array([mu*battery_times_mu_E ,mu*battery_times_mu_P,mu*battery_times_mu_pv],dtype= float),
              sigma = np.array([0.2, 0.2,0.2],dtype = float),
              cost_per_kWh_max=241/10,#$241/kWh/10yr*150yen/$*0.8
              cost_per_kW_max=372/10,#$372/kWh/10yr*150yen/$*0.8
              cost_per_kw_pv = 1080/10,#250/kW/10yr
              scheduling_decay= 0.995)

if __name__ == "__main__":
     seeds = [1,2,3,4,5]  
     battery_times_mu_pv_min = 1.0
     battery_times_mu_pv_max = 1.0
     solar_radiation_all = pd.read_csv("/home/students1/mantani/IEEE_TEMPR_value_stacking/data/caiso_renewables_hourly_solar_week.csv") 
     pv_actual_vals = solar_radiation_all.iloc[:, 1].to_numpy()/ 1000
     battery_times_mu_pv = battery_times_mu_pv_min
     mus = [max(pv_actual_vals)]  # 直接リストで指定

     while battery_times_mu_pv <= battery_times_mu_pv_max:
         tasks = []
         for mu in mus:
             for seed in seeds:
                 task = run_training.remote(seed,battery_times_mu_pv)
                 tasks.append(task)
            
         ray.get(tasks)
        
         battery_times_mu_pv += 0.5  # 1000ずつ増やす

