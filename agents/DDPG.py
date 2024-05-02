import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.set_device(0)

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action, hist_len):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(state_dim + 30, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, action_dim)

#         self.max_action = max_action
#         self.context = Context(hidden_sizes=[30],
#                                    input_dim=state_dim + action_dim + 1,
#                                    output_dim = 30, # hiddens_conext
#                                    only_concat_context = 3,
#                                    history_length = hist_len,
#                                    action_dim = action_dim,
#                                    obsr_dim = state_dim,
#                                    device = device
#                                    )
        
#     def forward(self, state, pre_infos = None, ret_context  = False):
        
#         combined = self.context(pre_infos)
#         state = torch.cat([state, combined], dim=-1)
        
#         a = F.relu(self.l1(state))
#         a = F.relu(self.l2(a))
#         a = self.max_action * torch.tanh(self.l3(a))
#         if ret_context == True:
#             return a, combined
#         else:
#             return a


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim,hist_len):
#         super(Critic, self).__init__()

#         # Q1 architecture
#         self.l1 = nn.Linear(state_dim + action_dim + 30, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 1)

#         # Q2 architecture
#         self.l4 = nn.Linear(state_dim + action_dim + 30, 256)
#         self.l5 = nn.Linear(256, 256)
#         self.l6 = nn.Linear(256, 1)

#         self.context = Context(hidden_sizes=[30],
#                                    input_dim=state_dim + action_dim + 1,
#                                    output_dim = 30, # hiddens_conext
#                                    only_concat_context = 3,
#                                    history_length = hist_len,
#                                    action_dim = action_dim,
#                                    obsr_dim = state_dim,
#                                    device = device
#                                    )
        
#     def forward(self, state, action, pre_info = None, ret_context = False):
#         sa = torch.cat([state, action], 1)

#         combined = self.context(pre_info)
#         sa = torch.cat([sa, combined], dim=-1)
        
#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)

#         q2 = F.relu(self.l4(sa))
#         q2 = F.relu(self.l5(q2))
#         q2 = self.l6(q2)
        
#         if ret_context == True:
#             return q1, q2, combined
#         else:
#             return q1, q2


#     def Q1(self, state, action, pre_info = None, ret_context = False):
#         sa = torch.cat([state, action], 1)
#         combined = self.context(pre_info)
#         sa = torch.cat([sa, combined], dim=-1)
        
#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)
#         if ret_context == True:
#             return q1, combined
#         else:
#             return q1
        
# class Context(nn.Module):
#     """
#       This layer just does non-linear transformation(s)
#     """
#     def __init__(self,
#                  hidden_sizes = [50],
#                  output_dim = None,
#                  input_dim = None,
#                  only_concat_context = 0,
#                  hidden_activation=F.relu,
#                  history_length = 1,
#                  action_dim = None,
#                  obsr_dim = None,
#                  device = 'cpu'
#                  ):

#         super(Context, self).__init__()
#         self.only_concat_context = only_concat_context
#         self.hid_act = hidden_activation
#         self.fcs = [] # list of linear layer
#         self.hidden_sizes = hidden_sizes
#         self.input_dim = input_dim
#         self.output_dim_final = output_dim # count the fact that there is a skip connection
#         self.output_dim_last_layer  = output_dim // 2
#         self.hist_length = history_length
#         self.device = device
#         self.action_dim = action_dim
#         self.obsr_dim = obsr_dim

#         #### build LSTM or multi-layers FF
#         if only_concat_context == 3:
#             # use LSTM or GRU
#             self.recurrent =nn.GRU(self.input_dim,
#                                self.hidden_sizes[0],
#                                bidirectional = False,
#                                batch_first = True,
#                                num_layers = 1)

#     def init_recurrent(self, bsize = None):
#         '''
#             init hidden states
#             Batch size can't be none
#         '''
#         # The order is (num_layers, minibatch_size, hidden_dim)
#         # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
#         #        torch.zeros(1, bsize, self.hidden_sizes[0]))
#         return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

#     def forward(self, data):
#         '''
#             pre_x : B * D where B is batch size and D is input_dim
#             pre_a : B * A where B is batch size and A is input_dim
#             previous_reward: B * 1 where B is batch size and 1 is input_dim
#         '''
#         previous_action, previous_reward, pre_x = data[0], data[1], data[2]
        
#         if self.only_concat_context == 3:
#             # first prepare data for LSTM
#             bsize, dim = previous_action.shape # previous_action is B* (history_len * D)
#             pacts = previous_action.view(bsize, -1, self.action_dim) # view(bsize, self.hist_length, -1)
#             prews = previous_reward.view(bsize, -1, 1) # reward dim is 1, view(bsize, self.hist_length, 1)
#             pxs   = pre_x.view(bsize, -1, self.obsr_dim ) # view(bsize, self.hist_length, -1)
#             pre_act_rew = torch.cat([pacts, prews, pxs], dim = -1) # input to LSTM is [action, reward]

#             # init lstm/gru
#             hidden = self.init_recurrent(bsize=bsize)

#             # lstm/gru
#             self.recurrent.flatten_parameters()
#             _, hidden = self.recurrent(pre_act_rew, hidden) # hidden is (1, B, hidden_size)
#             out = hidden.squeeze(0) # (1, B, hidden_size) ==> (B, hidden_size)

#             return out

#         else:
#             raise NotImplementedError

#         return None

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 40)
        self.l2 = nn.Linear(40, 30)
        self.l3 = nn.Linear(30, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) 
        return self.max_action * torch.sigmoid(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 40)
        self.l2 = nn.Linear(40 + action_dim, 30)
        self.l3 = nn.Linear(30, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG(object):
    
    def __init__(self, state_dim, action_dim, max_action, policy_noise =0.2, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.action_shape = action_dim
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)
        self.action_noise = policy_noise
        self.action_min = 0
        self.action_max = max_action

        self.discount = discount
        self.tau = tau


    def select_action(self, state, add_noise = True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
        # Add Gaussian noise to the action
            noise = np.random.normal(scale=self.action_noise, size=self.action_shape)
            action += noise
        return np.clip(action, self.action_min, self.action_max)


    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
