import copy
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils as torch_utils

from policy import Policy
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

class Trainer:
    def __init__(self, args, policies):
        self.args = args        
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.writer = SummaryWriter(log_dir = self.args.tensorboard_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           
        self.policies = policies
        self.target_policeis = copy.deepcopy(policies)
        
        self.params = []
        for agent_id, policy in enumerate(self.policies):        
            self.params += list(policy.model.parameters())
        
        self.target_params = []
        for agent_id, target_policy in enumerate(self.target_policeis):
            self.target_params += list(target_policy.model.parameters())        
        
        # define optimizer
        self.optimizer = torch.optim.Adam(params=self.params, lr = self.args.lr_policy)

    def train(self, q_values, rewards, next_q_values, time_step):        
        image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4                                
        
        q_values = torch.reshape(q_values, [self.args.batch_size, self.args.num_drones])
        next_q_values = torch.reshape(next_q_values, [self.args.batch_size, self.args.num_drones])          
        targets = rewards.unsqueeze(1) + self.args.gamma * next_q_values                                 
        
        self. optimizer.zero_grad()        
        loss = (targets - q_values).pow(2).mean()          
        loss.backward()
        torch_utils.clip_grad_norm_(self.params, 0.5)        
        self.optimizer.step()  
        self.soft_target_network_update()          
        
        # if time_step >=self.args.wait_steps and time_step % self.args.update_target_frequency == 0:
        #     self.update_target_network()
        #     print('=====[Target has been updated]======')
        
        if time_step % 300 == 0:
            self.writer.add_scalar('loss' , loss, int(time_step - 300 / 300))

    def get_samples(self, buffer, time_step):                
        self.buffer = buffer # size = (batch_size, num_agents, length_trajectories)
        image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4
        action_length = 1
        reward_length = 1
        q = torch.empty(self.args.batch_size, self.args.num_drones, self.args.action_shape).to(self.device)
        next_q = torch.empty(self.args.batch_size, self.args.num_drones, 1).to(self.device)
        chosen_q = torch.empty((self.args.batch_size, self.args.num_drones, 1)).type(torch.cuda.FloatTensor)              

        # create mini batch
        for agent_id in range(self.args.num_drones):
            batch = self.buffer[:, agent_id, :]
            batch = np.reshape(batch, [self.args.batch_size, -1]) # size = (batch, length_trajectories)

            # get states
            states = torch.from_numpy(batch[:, 0:image_length])
            states = Variable(states).type(torch.cuda.FloatTensor)    
            # calculate q values
            for policy in self.policies:
                q[:,agent_id, :] = policy.model(states)        
            # calculate actions
            actions = np.reshape(batch[:, image_length], [self.args.batch_size, 1])             
            actions = np.reshape(actions, [self.args.batch_size, 1])

            # calculate the chosen q_values            
            index = 0
            for action, q_value in zip(actions, q[:,agent_id,:]): 
                chosen_q[index, agent_id, :] = q_value[int(action)]
                index +=1 

            # get reward
            rewards = batch[:, image_length + action_length]
            rewards = Variable(torch.from_numpy(rewards)).type(torch.cuda.FloatTensor)
            # get next states
            next_states = Variable(torch.from_numpy(batch[:, image_length + action_length + reward_length:])).type(torch.cuda.FloatTensor)                  
            for target_policy in self.target_policeis:
                _next_q = target_policy.model(next_states).detach()                
                _next_q, _ = torch.max(_next_q, 1)                
                next_q[:, agent_id, :] = _next_q.unsqueeze(1)

        return [chosen_q, rewards, next_q]
    
    def update_target_network(self):
        self.target_params = copy.deepcopy(self.params)        

    def soft_target_network_update(self):
        self.target_params.data.copy_((1-self.args.tau) * self.target_params.data + self.args.tau * self.params.data)

    def save_model(self, time_step):
        num = str(time_step // self.args.save_rate)
        model_path = os.path.join(self.args.svae_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy_network.state_dict(), model_path + '/' + num + '_policy_params.pkl')        