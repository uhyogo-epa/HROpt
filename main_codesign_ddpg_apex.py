import numpy as np
import pandas as pd
import random
import torch
import ray
from datetime import datetime
from collections import deque
import copy
from torch.utils.tensorboard import SummaryWriter
from rl_env.codesign_lstm_year_multi import MultiyearEnv
from agent.codesign_ddpg_ApeX import CodesignDDPGApeXagent, Actor, Critic, ReplayMemory

@ray.remote
class Worker:
    def __init__(self, observation_space, action_space, rho_space, max_action, max_size,device, agent_idx,mode):
        self.agent_idx = agent_idx
        self.max_size = max_size
        self.device = device    
        self.env = MultiyearEnv(agent_idx,mode)
        self.agent = CodesignDDPGApeXagent(observation_space, action_space, rho_space, max_action)
        self.max_action   = max_action
        
    def run_episode(self, observation_space, action_space, rho_space, max_size, noise, gamma, max_step, weights_actor, weights_critic, mu, sigma, P_inv,update_cycles):
        # 重みの更新
        self.agent.actor.load_state_dict(weights_actor)
        self.agent.critic.load_state_dict(weights_critic)
        
        # 環境の初期化
        rho = np.random.normal(mu, sigma)
        rho = np.maximum(rho, 0.05)
        E_B, P_B, P_pv = rho
        obs = self.env.reset(E_B, P_B, P_pv, P_inv)
        
        done = False
        episode_reward = 0
        episode_revenue = 0
        local_buffer = []
        
        # 行動空間の次元を取得（action_spaceから取得するか、既存の変数があればそれを利用）
        action_dim = 5

        for t in range(max_step):
            # --- ここから修正部分 ---
            if update_cycles < 10:
                # 最初の10エピソードは完全にランダムな行動を選択
                action = self.max_action * np.random.rand(action_dim)
            else:
                action = self.agent.get_action(obs, rho)

                noise_sample = np.random.normal(
                   0, self.max_action * noise, size=action_dim
               )
                action = (action + noise_sample).clip(0.0, 1.0)

            next_state, reward, done, revenue = self.env.step(action)
            local_buffer.append((obs, action, next_state, reward, done, rho))

            obs = next_state
            episode_reward += reward
            episode_revenue += revenue
            
            if done:
                break
                
        return self.agent_idx, rho, episode_reward, local_buffer, episode_revenue
# ===============================
#  Learner (Remote Actor)
# ===============================
@ray.remote(num_cpus = 1)
class Learner:
    def __init__(self, observation_space, action_space, rho_space, max_action, device, tau, target_update_period, lr_actor,lr_critic, lr_mu, lr_sigma):
        self.agent = CodesignDDPGApeXagent(observation_space, action_space, rho_space, max_action)
        self.tau = tau
        self.target_update_period = target_update_period
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.replay_buffer = ReplayMemory(observation_space, action_space, rho_space, 500000)

        
    def get_weights(self):
        return self.agent.actor.state_dict(),self.agent.critic.state_dict()
    
    def add_memory(self, batch):
        self.replay_buffer.add_batch(batch)
    
    def learn(self):
        if self.replay_buffer.size < 20000:
            return self.get_weights()

        for _ in range(100):
            self.agent.update_step_from_buffer(self.replay_buffer)
            
        return self.get_weights()
    
    def update_phi(self, mu, sigma, rho_E_list, rho_P_list, rho_PV_list,income_list,
               cost_per_kWh, cost_per_kW, cost_per_PV):

        rhos_E    = np.array(rho_E_list[-10:])
        rhos_P    = np.array(rho_P_list[-10:])
        rhos_PV    = np.array(rho_PV_list[-10:])
        G = np.array(income_list[-10:]) - rhos_E * 1000 *cost_per_kWh - rhos_P * 1000* cost_per_kW - rhos_PV *1000 *cost_per_PV
        rhos = np.stack([rhos_E, rhos_P, rhos_PV], axis=1)
        # G       = q_max - rhos * battery_price  # Profit - battery_capacity * battery_price
        mu_grad =  (((rhos - mu) / (sigma ** 2)) * (G - G.mean())[:, None]).mean(axis=0)
        mu      = mu     + self.lr_mu * mu_grad

            # update sigma            
        sigma_grad = 0
        sigma = sigma + self.lr_sigma * sigma_grad
      
    
        return mu, sigma
    
    # def save_weights(self,ckpt_path):
        
    #     torch.save({'Q_net': self.Q.state_dict(),
    #                 'target-Q_net': self.Q_target.state_dict(),
    #                 }, 
    #     ckpt_path)


# ===============================
#  Tester (Remote Actor)
# ===============================
@ray.remote
class Tester:
   def __init__(self, observation_space, action_space, rho_space, max_action,device, agent_idx,mode, noise):
        self.agent = CodesignDDPGApeXagent(observation_space, action_space, rho_space, max_action)
        self.noise = noise
        self.agent_idx = agent_idx
        self.env = MultiyearEnv(agent_idx,mode)
        self.max_action = max_action
        self.agent = CodesignDDPGApeXagent(
          observation_space, action_space, rho_space, max_action
)
        self.device = device
        
   def run_episode(self, observation_space,action_space,rho_space,max_size, noise, gamma, max_step, weights_actor, weights_critic, mu, sigma,P_inv,update_cycles):
       self.agent.actor.load_state_dict(weights_actor)
       self.agent.critic.load_state_dict(weights_critic)

       rho = np.random.normal(mu, sigma)
       rho = np.maximum(rho,0.05)
       E_B, P_B, P_pv = rho
       obs = self.env.reset(E_B, P_B, P_pv,P_inv)
       done = False
       episode_reward = 0
       episode_revenue = 0
       local_buffer = []
       episode_reward_history         = []
       episode_revenue_energy_history = []
       episode_revenue_fcas_history   = []
       episode_capacity_payment_history = []
       episode_imbalance_penalty_history = []
       episode_degradation_cost_history = []
       episode_bidding_energy_history = []
       episode_bidding_reg_up_history = []
       episode_bidding_reg_dn_history = []
       episode_bidding_res_history    = []
       episode_bidding_reserve_history         = []
       action_dim = 5
       for t in range(max_step):
            # --- ここから修正部分 ---
           if update_cycles < 10:
                # 最初の10エピソードは完全にランダムな行動を選択
                action = self.max_action * np.random.rand(action_dim)
           else:
                action = self.agent.get_action(obs, rho)

                noise_sample = np.random.normal(
                    0, self.max_action * noise, size=action_dim
                )
                action = (action + noise_sample).clip(0.0, 1.0)

           next_state, reward, done, revenue = self.env.step(action)
           local_buffer.append((obs, action, next_state, reward, done,rho))
           obs = next_state
           episode_reward += reward
           episode_revenue += revenue
           episode_reward_history.append(reward)
           episode_bidding_energy_history.append(self.env.b_en)
           episode_bidding_reg_up_history.append(self.env.b_up)
           episode_bidding_reg_dn_history.append(self.env.b_dn)
           episode_bidding_res_history.append(self.env.b_res)
           episode_revenue_energy_history.append(self.env.revenue_energy)
           episode_revenue_fcas_history.append(self.env.revenue_fcas)
           episode_capacity_payment_history.append(self.env.capacity_payment)
           episode_degradation_cost_history.append(self.env.degradation_cost)
           episode_imbalance_penalty_history.append(self.env.imbalance_revenue)
           episode_bidding_reserve_history.append(self.env.b_res)
           if done:
               break

       return self.agent_idx, rho, episode_reward, local_buffer, episode_revenue, episode_bidding_energy_history,episode_bidding_reg_up_history,episode_bidding_reg_dn_history,episode_bidding_res_history, episode_reward_history, episode_revenue_energy_history,episode_revenue_fcas_history,episode_capacity_payment_history,episode_imbalance_penalty_history,episode_degradation_cost_history,episode_bidding_reserve_history
# ===============================
#  Main Training Loop
# ===============================
def main(num_cpus=12, gamma=0.99):
    random.seed(1)
    np.random.seed(1)
    mode = 'hybrid'
    # 環境の設定
    env = MultiyearEnv(0,mode)
    observation_space = env.observation_space
    action_space = env.action_space
    rho_space = env.rho_space
    max_action = 1
    Actor_learning_rate  = 1e-4
    Critic_learning_rate = 1e-4
    target_update_period = 4
    
    #ノイズの設定
    POLICY_NOISE      = 0.5
    NOISE_DECAY       = 0.9995
    NOISE_MIN         = 0.01
    
    REPLAY_MEMORY_SIZE = 100000
    tau = 5e-3
    max_step = 8760
    num_updates  = 5000
    device = torch.device("cpu")

    ##############################
    # Codesing parameters
    ##############################ra-ni
    learning_rate_mu    = 0
    learning_rate_sigma = 0
    min_epi_codesign = 20000
    device = torch.device("cpu")
    
    # data for co-design
    battery_times_mu_E = 2.0
    battery_times_mu_P = 0.5
    battery_times_mu_pv = 1.0
    P_inv = 10
    cost_per_kWh_max=241/10#$241/kWh/10yr*150yen/$*0.8
    cost_per_kW_max=372/(10*4)#$372/kWh/10yr*150yen/$*0.8
    cost_per_kw_pv = 100/10
    mu    = np.array([P_inv*battery_times_mu_E ,P_inv*battery_times_mu_P,P_inv*battery_times_mu_pv],dtype= float)
    sigma = np.array([0.00001, 0.0000001,0.000001],dtype = float)

    
    income_list = [] ############# TODO: reward -> income
    reward_list = []
    rho_E_list    = []
    rho_P_list    = []
    rho_PV_list    = []
    history = []
    bidding_energy_history = []
    bidding_reg_up_history = []
    bidding_reg_dn_history = []
    bidding_res_history    = []
    revenue_energy_history = []
    revenue_fcas_history   = []
    capacity_payment_history = []
    imbalance_revenue_history = []
    degradation_cost_history = []
    bidding_reserve_history             = []
    log_dir = 'codesign_apex_logs/test_run_'+datetime.now().strftime('%m%d%H%M')
    writers  = SummaryWriter(log_dir=log_dir)
    
    # replay_memory = ReplayMemory(observation_space, action_space, rho_space, REPLAY_MEMORY_SIZE)
    # Ray 初期化
    ray.init(ignore_reinit_error=True)

    # Worker と Learner のセットアップ
    workers = [Worker.remote(observation_space, action_space, rho_space, max_action,REPLAY_MEMORY_SIZE,device, agent_idx,mode) for agent_idx in range(num_cpus)]
    learner = Learner.remote(observation_space, action_space, rho_space, max_action,device, tau, target_update_period, Actor_learning_rate, Critic_learning_rate, learning_rate_mu,learning_rate_sigma)
    noise = POLICY_NOISE
    noise_tester = 0
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    current_weights_actor,current_weights_critic = ray.get(learner.get_weights.remote())    
   
    tester = Tester.remote( observation_space, action_space, rho_space, max_action,device, "all", mode, noise_tester)
    update_cycles = 1
    actor_cycles = 0
    
    # 初期エピソードの収集
    worker_results = [
    worker.run_episode.remote(
        observation_space, action_space, rho_space, 
        REPLAY_MEMORY_SIZE,
        noise, gamma, max_step, 
        current_weights_actor, current_weights_critic, 
        mu, sigma, P_inv,update_cycles # ここに P_inv があるか確認
        
    ) for worker in workers
]
    for _ in range(30):
        finished, worker_results = ray.wait(worker_results, num_returns=1)
        agent_idx, rho, episode_reward,local_buffer,episode_revenue = ray.get(finished[0])
        learner.add_memory.remote(local_buffer)
        worker_results.append(
        workers[agent_idx].run_episode.remote(
            observation_space, action_space, rho_space, 
            REPLAY_MEMORY_SIZE,
            noise, gamma, max_step, 
            current_weights_actor, current_weights_critic, 
            mu, sigma, P_inv,update_cycles # ここにも P_inv を追加
        )
    )
    # 学習開始
    # minibatches = [ReplayMemory.sample.remote() for _ in range(24 * 7)]
    wip_learner = learner.learn.remote()
    wip_tester = tester.run_episode.remote(observation_space,action_space,rho_space,  REPLAY_MEMORY_SIZE, noise_tester, gamma, max_step, current_weights_actor, current_weights_critic, mu, sigma,P_inv,update_cycles)
    

    while update_cycles <= num_updates:
        # workerによるデータの収集
        actor_cycles += 1
        finished, remaining = ray.wait(worker_results, num_returns=1)
        worker_results = remaining
        # agent_idx, rho, episode_reward, local_buffer, episode_revenue = ray.get(finished[0])
        
        agent_idx, rho, episode_reward,local_buffer,episode_revenue = ray.get(finished[0])
        learner.add_memory.remote(local_buffer)
        
        reward_list.append(episode_reward)
        income_list.append(episode_revenue)
        rho_E_list.append(rho[0])
        rho_P_list.append(rho[1])
        rho_PV_list.append(rho[2])

        # writers.add_scalar('Rewards', episode_reward, update_cycles)
        #writers.add_scalar('Mu', mu, update_cycles)
        
        worker_results.extend([workers[agent_idx].run_episode.remote(observation_space,action_space,rho_space, REPLAY_MEMORY_SIZE,noise,  gamma, max_step, current_weights_actor, current_weights_critic, mu, sigma,P_inv,update_cycles)])   

        # Learnerのタスク完了判定とDDPGの更新
        finished_learner, _ = ray.wait([wip_learner], timeout=0)
        if finished_learner:
            update_cycles += 1
            print("Actorが遷移をReplayに渡した回数：", actor_cycles)
            actor_cycles = 0

            # DRQNの更新の開始
            new_weights_actor, new_weights_critic     = ray.get(finished_learner[0])
            current_weights_actor = ray.put(new_weights_actor)
            current_weights_critic = ray.put(new_weights_critic)
            
            # minibatches = [ReplayMemory.sample() for _ in range(24*7)]
            wip_learner     = learner.learn.remote()
            
            if update_cycles % 5 == 0:
                test_score = ray.get(wip_tester)
                print(update_cycles, test_score[1], test_score[2],test_score[4])
                history.append((update_cycles-5, test_score))
                wip_tester = tester.run_episode.remote(observation_space,action_space,rho_space, REPLAY_MEMORY_SIZE, noise_tester, gamma, max_step, current_weights_actor,current_weights_critic, mu, sigma,P_inv,update_cycles)
                # mu = test_score[1]
                # episode_reward = test_score[2]  
                # writers.add_scalar('wip_tester/rho', rho, update_cycles)
                episode_reward = test_score[2]
                episode_revenue = test_score[4]
                episode_bidding_energy = test_score[5]
                episode_bidding_reg_up = test_score[6]
                episode_bidding_reg_dn = test_score[7]
                episode_bidding_res    = test_score[8]
                episode_step_reward    = test_score[9]
                episode_revenue_energy = test_score[10]
                episode_revenue_fcas   = test_score[11]
                episode_capacity_payment = test_score[12]
                episode_imbalance_penalty = test_score[13]
                episode_degradation_cost = test_score[14]
                episode_bidding_reserve         = test_score[15]
                writers.add_scalar('wip_tester/reward', episode_reward, update_cycles)
                writers.add_scalar('wip_tester/revenue', episode_revenue, update_cycles)
                bidding_energy_history.append(episode_bidding_energy)
                bidding_reg_up_history.append(episode_bidding_reg_up)
                bidding_reg_dn_history.append(episode_bidding_reg_dn)
                bidding_res_history.append(episode_bidding_res)
                revenue_energy_history.append(episode_revenue_energy)
                revenue_fcas_history.append(episode_revenue_fcas)
                imbalance_revenue_history.append(episode_imbalance_penalty)
                capacity_payment_history.append(episode_capacity_payment)
                degradation_cost_history.append(episode_degradation_cost)
                bidding_reserve_history.append(episode_bidding_reserve)
                # writers.add_scalar('wip_tester/Deviation_penalty', sum(test_score[4]), update_cycles)
                # writers.add_scalar('wip_tester/Degradation', sum(test_score[5]), update_cycles)
               
                if update_cycles % num_updates ==  0:
                    reward_df = pd.DataFrame(episode_step_reward)
                    reward_df.to_csv(f'ddpg_apex_results/episode_reward_{current_time}.csv',index= False)
                    bidding_energy_df = pd.DataFrame(episode_bidding_energy)
                    bidding_energy_df.to_csv(f'ddpg_apex_action/episode_bidding_energy_{current_time}.csv',index= False)
                    bidding_reg_up_df = pd.DataFrame(episode_bidding_reg_up)
                    bidding_reg_up_df.to_csv(f'ddpg_apex_action/episode_bidding_reg_up_{current_time}.csv',index= False)
                    bidding_reg_dn_df = pd.DataFrame(episode_bidding_reg_dn)
                    bidding_reg_dn_df.to_csv(f'ddpg_apex_action/episode_bidding_reg_dn_{current_time}.csv',index= False)
                    bidding_reserve_df = pd.DataFrame(episode_bidding_reserve)
                    bidding_reserve_df.to_csv(f'ddpg_apex_action/episode_bidding_reserve_{current_time}.csv',index= False)                 
                    revenue_energy_df = pd.DataFrame(episode_revenue_energy)
                    revenue_energy_df.to_csv(f'ddpg_apex_results/episode_revenue_energy_{current_time}.csv',index= False)
                    revenue_fcas_df = pd.DataFrame(episode_revenue_fcas)
                    revenue_fcas_df.to_csv(f'ddpg_apex_results/episode_revenue_fcas_{current_time}.csv',index= False)
                    imbalance_revenue_df = pd.DataFrame(episode_imbalance_penalty)
                    imbalance_revenue_df.to_csv(f'ddpg_apex_results/episode_imbalance_{current_time}.csv',index= False)
                    capacity_payment_df = pd.DataFrame(episode_capacity_payment)
                    capacity_payment_df.to_csv(f'ddpg_apex_results/episode_capacity_{current_time}.csv',index= False)
                    degradation_cost_df = pd.DataFrame(episode_degradation_cost)
                    degradation_cost_df.to_csv(f'ddpg_apex_results/episode_degradation_{current_time}.csv',index= False)
            
            # # Phiの更新
            if update_cycles > min_epi_codesign:
                if update_cycles % 50 == 0:
            
                    mu, sigma = ray.get(
                        learner.update_phi.remote(
                            mu,
                            sigma,
                            rho_E_list[-50:],
                            rho_P_list[-50:],
                            rho_PV_list[-50:],
                            income_list[-50:],
                            cost_per_kWh_max,
                            cost_per_kW_max,
                            cost_per_kw_pv
                        )
                    )
            
                    writers.add_scalar('phi/mu_E', mu[0], update_cycles)
                    writers.add_scalar('phi/mu_P', mu[1], update_cycles)
                    writers.add_scalar('phi/mu_PV', mu[2], update_cycles)
            
           
                        # update epsilon
            noise = max(NOISE_MIN, noise *NOISE_DECAY ) # Linear annealing
            writers.add_scalar('noise', noise, update_cycles)
            
            # penalty_history.append(episode_penalty_history)
            # battery_penalty_history.append(episode_battery_penalty_history)
            #save agent
            # if update_cycles % 5000 == 0:
            #     ckpt_path = log_dir + f'/agent-{update_cycles}.pth'
            #     learner.save_weights.remote(ckpt_path)
                               
    ray.shutdown()


if __name__ == "__main__":
    main()
