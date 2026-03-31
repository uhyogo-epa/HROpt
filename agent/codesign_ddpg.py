import numpy as np
import copy
from tqdm import tqdm  # progress bar
import torch.nn.functional as F
import pandas as pd
from   datetime import datetime
import torch
import torch.nn as nn
from   torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")

# DDPG Agent class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, rho_dim, max_action):
        super(Actor, self).__init__()
        self.hidden_space = 64
        self.l1 = nn.Linear(state_dim+rho_dim, self.hidden_space)
        self.l2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.l3 = nn.Linear(self.hidden_space, action_dim)		
        self.max_action = max_action
        
    def forward(self, state,rho):
        x = torch.cat([state, rho], dim=-1)
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        raw_action = self.max_action * torch.tanh(self.l3(a))
        action = 0.5 * (raw_action + 1.0)
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,rho_dim):
        super(Critic, self).__init__()
        self.hidden_space = 64
        self.l1 = nn.Linear(state_dim + action_dim + rho_dim, self.hidden_space)
        self.l2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.l3 = nn.Linear(self.hidden_space, 1)
    def forward(self, state, action,rho):
        q1 = torch.relu(self.l1(torch.cat([state, action,rho], 1)))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
###########################################################################
# Reply buffer
###########################################################################
class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, rho_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # buffers
        self.state      = np.zeros((max_size, state_dim))
        self.action     = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))
        self.rho        = np.zeros((max_size, 3))#次元によって変更
        
    def add(self, state, action, next_state, reward, done,rho):
        # buffering
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.rho[self.ptr] = rho
        
        # move pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
            torch.FloatTensor(self.rho[ind]).to(device)
            )
    def __len__(self):
        return self.size
    
class CodesignDDPGagent:
    def __init__(self, state_dim, action_dim, rho_dim, max_action,
                 ACTOR_LEARN_RATE   = 1e-4,
                 CRITIC_LEARN_RATE  = 1e-4,
         		 DISCOUNT           = 0.9,
                 REPLAY_MEMORY_SIZE = 5000,
                 REPLAY_MEMORY_MIN  = 100,
                 MINIBATCH_SIZE     = 32,
                 EXPL_NOISE        = 0.15,      # Std of Gaussian exploration noise
                 TAU                = 0.005,
                 POLICY_NOISE       = 0.5,
                 NOISE_DECAY        = 1,
                 NOISE_CLIP         = 0.5,
                 NOISE_MIN          = 0.1,
                 POLICY_FREQ        = 4):
        
        self.actor = Actor(state_dim, action_dim, rho_dim,max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = ACTOR_LEARN_RATE)
        self.critic = Critic(state_dim, action_dim,rho_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = CRITIC_LEARN_RATE)
        self.max_action   = max_action
        
        # Replay Memory
        self.replay_memory = ReplayMemory(state_dim, action_dim,rho_dim, REPLAY_MEMORY_SIZE)
        self.REPLAY_MEMORY_MIN = REPLAY_MEMORY_MIN
        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        self.action_dim   = action_dim
        self.max_action   = max_action
        self.discount     = DISCOUNT
        self.EXPL_NOISE   = EXPL_NOISE
        self.tau          = TAU
        
        #noise parameter
        self.policy_noise = POLICY_NOISE
        self.noise_decay  = NOISE_DECAY
        self.noise_clip   = NOISE_CLIP
        self.noise_min    = NOISE_MIN
        self.policy_freq  = POLICY_FREQ
        
        self.policy_update_cnt = 0
        
    def get_action(self, state,rho):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        rho = torch.FloatTensor(rho.reshape(1, -1)).to(device)
        return self.actor(state,rho).cpu().data.numpy().flatten()
    
    def get_value(self, state, rho):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            rho = torch.FloatTensor(rho.reshape(1, -1)).to(device)
            action = self.actor(state, rho)
            value = self.critic(state, action, rho)
        return value.cpu().numpy().flatten()

    def train(self, env,
              mode,
              EPISODES      = 50,
              LOG_DIR       = None,
              SHOW_PROGRESS = True,
              SAVE_AGENTS   = True,
              SAVE_FREQ     = 500,
              RESTART_EP    = None,  
              seed = 1,
              learning_rate_mu    = 1e-6,
              learning_rate_sigma = 0,
              min_epi_codesign    = 300,
              pv_inv = 1,
              mu    = 0.1,
              sigma = 0.2,
              cost_per_kWh_max=241/10*0.8,#$241/kWh/10yr*150yen/$*0.8
              cost_per_kW_max=372/(10*4)*0.8,#$372/kWh/10yr*150yen/$*0.8
              cost_per_kw_pv_max = 150,
              scheduling_decay= 0.995):
    
       ######################################
       # Prepare log writer
       ######################################
       if LOG_DIR:
           summary_dir    = LOG_DIR
           summary_writer = SummaryWriter(log_dir=summary_dir)
       if LOG_DIR and SHOW_PROGRESS:
           print(f'Progress recorded: {summary_dir}')
           print(f'---> $tensorboard --logdir {summary_dir}')
           
       ##########################################
       # Define iterator for training loop
       ##########################################
       start = 0 if RESTART_EP == None else RESTART_EP
       if SHOW_PROGRESS:
           iterator = tqdm(range(start, EPISODES), ascii=True, unit='episodes')
       else:
           iterator = range(start, EPISODES)
       
       current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
       reward_list  = []
       revenue_list = []
       rho_E_list   = []
       rho_P_list   = []
       rho_I_list   = []
       b_res_history = []
       b_up_history = []
       b_dn_history = []
       b_en_history = []
       soc_history  = []
       p_pred_history  = []
       a_imb_history = []
       E_disE_history = []
       E_short_history = []
       scheduling_rate = 1
       for episode in iterator:
            if episode < min_epi_codesign:
                cost_per_kW = 0
                cost_per_kWh = 0
                cost_per_PV = 0
                
              # battery_price = battery_price_max*(1-np.exp(-scheduling_rate*(episode-300)))
            elif episode < 7000:
                 cost_per_kWh = cost_per_kWh_max* (1-scheduling_rate)             
                 cost_per_kW = cost_per_kW_max* (1-scheduling_rate)
                 cost_per_PV = cost_per_kw_pv_max* (1-scheduling_rate)
                 scheduling_rate = scheduling_rate*scheduling_decay
            else:
                 cost_per_kWh = cost_per_kWh_max
                 cost_per_kW  = cost_per_kW_max
                 cost_per_PV = cost_per_kw_pv_max
             
            is_done = False
            rho_E = max(0.05, np.random.normal(mu[0], sigma[0]))
            rho_P = max(0.05, np.random.normal(mu[1], sigma[1]))
            rho_I = max(0.05, np.random.normal(mu[2], sigma[2]))
            obs = env.reset(rho_E,rho_P,rho_I,pv_inv)
            rho = np.array([rho_E,rho_P,rho_I])
            episode_reward = 0
            episode_revenue = 0
            episode_reward_discount = 0
            episode_b_res = []  # List to collect bidding data
            episode_b_up = []
            episode_b_dn = []
            episode_a_imb = []
            episode_E_disE = []
            episode_E_short = []
            episode_b_en = []
            episode_soc = []
            episode_p_pred = []
            
            torch.set_grad_enabled(True)
            #######################################################
            # Iterate until episode ends
            #######################################################
            while not is_done:
                # get action
                if len(self.replay_memory) < self.REPLAY_MEMORY_MIN:
                    action = self.max_action * np.random.rand(self.action_dim)
                else:
                    noise = np.random.normal(0, self.max_action * self.EXPL_NOISE, size=self.action_dim)
                    action = (self.get_action(obs,rho)+noise).clip(-self.max_action, self.max_action)
                
                # make a step
                next_state, reward, is_done,revenue = env.step(action)
                episode_reward += reward
                episode_b_res.append(env.b_res)  # Collect bidding data
                episode_b_up.append(env.b_up)
                episode_b_dn.append(env.b_dn)
                episode_a_imb.append(env.a_imb)
                episode_E_disE.append(env.E_disE)
                episode_E_short.append(env.E_short)
                episode_b_en.append(env.b_en)
                episode_soc.append(env.soc)
                episode_p_pred.append(env.p_pred)
                
                with torch.no_grad():
                    q_values= self.get_value(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),torch.tensor(rho).float().unsqueeze(0).unsqueeze(0))
                    
                    

                # train Q network
                self.replay_memory.add(obs, action, next_state, reward, is_done,rho)
                if episode > 20:
                    self.update_step()

                # update current state
                obs = next_state
                episode_reward += reward
                episode_reward_discount = reward + self.discount*episode_reward_discount
                episode_revenue += revenue
                         
                if is_done:
                    break
            
            
            # bidding_history.append(episode_bidding)   
            
            b_res_history.append(episode_b_res)  # Save episode bidding data
            b_up_history.append(episode_b_up)
            b_dn_history.append(episode_b_dn)
            a_imb_history.append(episode_a_imb)
            E_disE_history.append(episode_E_disE)
            E_short_history.append(episode_E_short)
            b_en_history.append(episode_b_en)
            soc_history.append(episode_soc)
            p_pred_history.append(episode_p_pred)
            rho_E_list.append(rho_E)
            rho_P_list.append(rho_P)
            rho_I_list.append(rho_I)
            reward_list.append(episode_reward)
            revenue_list.append(episode_revenue)
            self.EXPL_NOISE *= self.noise_decay 
            self.EXPL_NOISE  = max(self.EXPL_NOISE,self.noise_min)
            
            batch_size_codesign = 100
            if episode >= min_epi_codesign:
                 if episode % batch_size_codesign ==0:
            
                    rhos_E    = np.array(rho_E_list[-batch_size_codesign:])
                    rhos_P    = np.array(rho_P_list[-batch_size_codesign:])
                    rhos_I    = np.array(rho_I_list[-batch_size_codesign:])
                    G = np.array(revenue_list[-batch_size_codesign:])*365/7 - rhos_E * 1000 *cost_per_kWh - rhos_P * 1000* cost_per_kW - rhos_I *1000 *cost_per_PV
                    rhos = np.stack([rhos_E, rhos_P, rhos_I], axis=1)
                    # G       = q_max - rhos * battery_price  # Profit - battery_capacity * battery_price
                    mu_grad =  (((rhos - mu) / (sigma ** 2)) * (G - G.mean())[:, None]).mean(axis=0)
                    mu      = mu     + learning_rate_mu * mu_grad

        
                    # update sigma            
                    sigma_grad = 0
                    sigma = sigma + learning_rate_sigma * sigma_grad
                    #  (rho_tensor - mu) ** 2  - sigma ** 2) / (sigma ** 3)
            b_res_history.append(episode_b_res)
            b_up_history.append(episode_b_up)
            b_dn_history.append(episode_b_dn)
            a_imb_history.append(episode_a_imb)
            E_disE_history.append(episode_E_disE)
            E_short_history.append(episode_E_short)
            b_en_history.append(episode_b_en)
            soc_history.append(episode_soc)
            p_pred_history.append(episode_p_pred)
            
            print(f"Episode {episode + 1}: Reward : {episode_reward}")
            print(f"Episode {episode + 1}: Reward_discount : {episode_reward_discount}")
            print(f"Episode {episode + 1}: Mu:{mu}")
            
            ###################################################
            # Log
            ###################################################
            if LOG_DIR:
                summary_writer.add_scalar("Episode Reward", episode_reward, episode)
                summary_writer.add_scalar("Episode Revenue", episode_revenue, episode)
                summary_writer.add_scalar("Episode Q0",     q_values,     episode)
                summary_writer.add_scalar('E_B', mu[0], episode)
                summary_writer.add_scalar('P_B', mu[1], episode)
                summary_writer.add_scalar('P_pv', mu[2], episode)
                summary_writer.add_scalar('noise', self.EXPL_NOISE, episode)
                
                summary_writer.flush()
       summary_writer.close()
            
       b_res_df = pd.DataFrame(b_res_history)
       b_up_df = pd.DataFrame(b_up_history)
       b_dn_df = pd.DataFrame(b_dn_history)
       a_imb_df = pd.DataFrame(a_imb_history)
       E_disE_df = pd.DataFrame(E_disE_history)
       E_short_df = pd.DataFrame(E_short_history)
       b_en_df = pd.DataFrame(b_en_history)
       soc_df  = pd.DataFrame(soc_history)
       p_pred_df  = pd.DataFrame(p_pred_history)
        
        # Save each DataFrame to CSV files
       b_res_df.to_csv(f'action/episode_b_res_{mode}_{seed}_{current_datetime}.csv', index=False)
       b_up_df.to_csv(f'action/episode_b_up_{mode}_{seed}_{current_datetime}.csv', index=False)
       b_dn_df.to_csv(f'action/episode_b_dn_{mode}_{seed}_{current_datetime}.csv', index=False)
       a_imb_df.to_csv(f'action/episode_a_imb_{mode}_{seed}_{current_datetime}.csv', index=False)
       E_disE_df.to_csv(f'action/episode_E_disE_{mode}_{seed}_{current_datetime}.csv', index=False)
       E_short_df.to_csv(f'action/episode_E_short_{mode}_{seed}_{current_datetime}.csv', index=False)
       b_en_df.to_csv(f'action/episode_b_en_{mode}_{seed}_{current_datetime}.csv', index=False)
       soc_df.to_csv(f'action/episode_soc_{mode}_{seed}_{current_datetime}.csv', index=False)        
       p_pred_df.to_csv(f'action/episode_p_pred_{mode}_{seed}_{current_datetime}.csv', index=False)
                        
                # if SAVE_AGENTS and episode % SAVE_FREQ == 0:
                #     ckpt_path = summary_dir + f'/agent-{episode}'
                #     torch.save({'actor-weights':  self.actor.state_dict(),
                #                 'critic-weights': self.critic.state_dict(),
                #                 }, ckpt_path)
                       
    ####################################################################
    # Update of actor and critic
    ####################################################################
    def update_step(self):
        ###############################################
        # Calculate Critic loss
        ###############################################
        
        # Sample replay buffer
        state, action, next_state, reward, not_done,rho = self.replay_memory.sample(self.MINIBATCH_SIZE)
        # Calculate target
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
			
            next_action = (
                self.actor_target(next_state,rho) + noise
                ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action,rho)
            target_Q = reward + not_done * self.discount * target_Q
            
        # Compute critic loss
        current_Q = self.critic(state, action,rho)
        critic_loss = F.mse_loss(current_Q, target_Q)
       
        #####################################
        # Update neural network weights
        #####################################
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        ###############################################
        # Delayed policy updates
        ###############################################
        self.policy_update_cnt = (self.policy_update_cnt + 1) % self.policy_freq
        if self.policy_update_cnt == 0:

            state_detached = state.detach()
            rho_detached   = rho.detach()

            actor_action = self.actor(state_detached, rho_detached)

            actor_loss = -self.critic(
                state_detached,
                actor_action,
                rho_detached
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
