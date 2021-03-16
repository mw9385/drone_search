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
    def __init__(self, args, agent_id, policy):
        self.args = args
        self.agent_id = agent_id
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.writer = SummaryWriter(log_dir = self.args.tensorboard_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           

        # allocate local and target policy
        self.policy_network = policy.to(self.device)
        self.policy_target_network = copy.deepcopy(policy).to(self.device)

        # store parameters
        self.params = list(self.policy_network.parameters())        
        self.target_params = list(self.policy_target_network.parameters())
        # define optimizer
        self.optimizer = torch.optim.Adam(params=self.params, lr = self.args.lr_policy)

    def train(self, q_values, rewards, next_q_values, time_step, mode):        
        image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4                                
        
        q_values = q_values
        next_q_values = next_q_values            
        targets = rewards + self.args.gamma * next_q_values                                

        self. optimizer.zero_grad()
        loss = (targets - q_values).pow(2).mean()
        loss.backward()
        torch_utils.clip_grad_norm_(self.params, 0.5)        
        self.optimizer.step()            
        
        if time_step >=self.args.wait_steps and time_step % self.args.update_target_frequency == 0:
            self.update_target_network()
        
        if time_step % 300 == 0:
            self.writer.add_scalar('loss_%i' % self.agent_id , loss, int(time_step - 300 / 300))

    def get_samples(self, buffer, time_step):                
        self.buffer = buffer
        image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4
        action_length = 1
        reward_length = self.args.num_drones

        # create mini batch
        batch = self.buffer[:, self.agent_id, :]
        batch = np.reshape(batch, [self.args.batch_size, -1])
        # get states
        states = torch.from_numpy(batch[:, 0:image_length])
        states = Variable(states).type(torch.cuda.FloatTensor)    
        # calculate q values
        q_values = self.policy_network(states)
        
        # calculate actions
        actions = np.reshape(batch[:, image_length], [self.args.batch_size, 1])             
        actions = np.reshape(actions, [self.args.batch_size, 1])

        # calculate the chosen q_values
        chosen_q_values = torch.empty((self.args.batch_size)).type(torch.cuda.FloatTensor)              
        index = 0
        for action, q_value in zip(actions, q_values):                            
            chosen_q_values[index] = q_value[int(action)]
            index +=1 
        # get reward
        rewards = batch[:, image_length + action_length: image_length + action_length + reward_length]
        rewards = Variable(torch.from_numpy(rewards)).type(torch.cuda.FloatTensor)
        # get next states
        next_states = Variable(torch.from_numpy(batch[:, image_length + action_length + reward_length:])).type(torch.cuda.FloatTensor)                  
        next_q_values = self.policy_target_network(next_states).detach()   
        max_next_q_values, _ = torch.max(next_q_values, 1)         

        return [chosen_q_values, rewards, max_next_q_values]
    
    def update_target_network(self):
        self.target_params = copy.deepcopy(self.params)        

    def save_model(self, time_step):
        num = str(time_step // self.args.save_rate)
        model_path = os.path.join(self.args.svae_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy_network.state_dict(), model_path + '/' + num + '_policy_params.pkl')    

    def save_model(self, time_step):
        num = str(time_step // self.args.save_rate)
        model_path = os.path.join(self.args.svae_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy_network.state_dict(), model_path + '/' + num + '_policy_params.pkl')