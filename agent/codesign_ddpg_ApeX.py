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
        self.hidden_space = 128
        self.l1 = nn.Linear(state_dim+rho_dim, self.hidden_space)
        self.l2 = nn.Linear(self.hidden_space, self.hidden_space)
        #self.l3 = nn.Linear(self.hidden_space, self.hidden_space)
        self.lo = nn.Linear(self.hidden_space, action_dim)		
        self.max_action = max_action
        
    def forward(self, state,rho):
        x = torch.cat([state, rho], dim=-1)
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        #a = F.relu(self.l3(a))
        raw_action = self.max_action * torch.tanh(self.lo(a))
        action = 0.5 * (raw_action + 1.0)
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,rho_dim):
        super(Critic, self).__init__()
        self.hidden_space = 128
        self.l1 = nn.Linear(state_dim + action_dim + rho_dim, self.hidden_space)
        self.l2 = nn.Linear(self.hidden_space, self.hidden_space)
        #self.l3 = nn.Linear(self.hidden_space, self.hidden_space)
        self.lo = nn.Linear(self.hidden_space, 1)
    def forward(self, state, action,rho):
        q1 = torch.relu(self.l1(torch.cat([state, action,rho], 1)))
        q1 = torch.relu(self.l2(q1))
        #q1 = torch.relu(self.l3(q1))
        q1 = self.lo(q1)
        return q1
    
###########################################################################
# Reply buffer
###########################################################################
class ReplayMemory:
    def __init__(self, state_dim, action_dim, rho_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.rho = np.zeros((max_size, rho_dim))

    def add(self, state, action, next_state, reward, done, rho):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.rho[self.ptr] = rho

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, batch):
        for data in batch:
            s, a, ns, r, d, rho = data
            self.add(s, a, ns, r, d, rho)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
            torch.FloatTensor(self.rho[ind]).to(device),
        )
    
class CodesignDDPGApeXagent:
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
              
    ####################################################################
    # Update of actor and critic
    ####################################################################
    def update_step_from_buffer(self, buffer):
        state, action, next_state, reward, not_done, rho = buffer.sample(
            self.MINIBATCH_SIZE
        )
    
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
    
            next_action = (
                self.actor_target(next_state, rho) + noise
            ).clamp(0.0, self.max_action)
    
            target_Q = self.critic_target(
                next_state, next_action, rho
            )
    
            target_Q = reward + not_done * self.discount * target_Q
    
        current_Q = self.critic(state, action, rho)
        critic_loss = F.mse_loss(current_Q, target_Q)
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
            
        # Actor update
        actor_action = self.actor(state, rho)
        actor_loss = -self.critic(state, actor_action, rho).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
    
        # soft update
        for param, target_param in zip(
            self.actor.parameters(),
            self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
