import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def _weight_initialization(self, layer):
        torch.nn.init.xavier_uniform_(layer)
    
    def __init__(self, args, agent_id):                  
        super(Actor, self).__init__()
        self.args = args
        self.max_action = 3
        self.image_dims = self.args.obs_shape * self.args.obs_shape * 3 + 4        
        self.hidden_nodes = 128

        # inputs: sensor image + closest_agent_id, closest_agnet_vector
        self.fc1 = nn.Linear(self.image_dims, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes)        
        self.fc3 = nn.Linear(self.hidden_nodes, self.args.action_shape)
        
        #initialization
        self._weight_initialization(self.fc1.weight)
        self._weight_initialization(self.fc2.weight)
        self._weight_initialization(self.fc3.weight)        

    def forward(self, x):     
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))              
        x = F.softmax(self.fc3(x)) 
        return x

class Critic(nn.Module):
    def _weight_initialization(self, layer):
        torch.nn.init.xavier_uniform_(layer)

    def __init__(self, args):        
        super(Critic, self).__init__()
        self.hidden_nodes = 128
        self.args = args        
        self.image_dims = self.args.obs_shape * self.args.obs_shape * 3 + 4                  
        # for image
        self.fc1 = nn.Linear(self.image_dims * self.args.num_drones + self.args.num_drones * self.args.action_shape, self.hidden_nodes)         
        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes)        
        self.fc3 = nn.Linear(self.hidden_nodes,  self.hidden_nodes)
        self.q_out = nn.Linear(self.hidden_nodes, 1)

        self._weight_initialization(self.fc1.weight)
        self._weight_initialization(self.fc2.weight)
        self._weight_initialization(self.fc3.weight)        
        self._weight_initialization(self.q_out.weight)        

    def forward(self, state, action):        
        # resize aciton and state
        state_size = state.shape[2]        
        action_size = action.shape[2]
        s = torch.reshape(state, (self.args.batch_size, state_size * self.args.num_drones))
        a = torch.reshape(action, (self.args.batch_size, self.args.num_drones * self.args.action_shape))  
        x = torch.cat([s, a], dim=1)                        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)
        q_value = self.q_out(x)        
        
        return q_value
