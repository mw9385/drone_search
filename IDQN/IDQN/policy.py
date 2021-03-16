import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class Policy(nn.Module):

    def _weight_initialization(self, layer):
        torch.nn.init.xavier_uniform_(layer)
        # torch.nn.init.kaiming_uniform_(layer)
    
    def __init__(self, args, agent_id):               
        super(Policy, self).__init__()
        self.args = args
        self.agent_id = agent_id
        self.image_dims = self.args.obs_shape * self.args.obs_shape*3 + 4
        self.hidden_nodes = 64

        # inputs: sensor image + closest_agent_id, closest_agnet_vector        
        self.fc1 = nn.Linear(self.image_dims, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.value_linear = nn.Linear(self.hidden_nodes, self.hidden_nodes) 
        self.value_out = nn.Linear(self.hidden_nodes, 1)
        self.advantage_linear = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.advatage_out = nn.Linear(self.hidden_nodes, self.args.action_shape)

        # initialization
        self._weight_initialization(self.fc1.weight)
        self._weight_initialization(self.fc2.weight)
        self._weight_initialization(self.value_linear.weight)
        self._weight_initialization(self.value_out.weight)
        self._weight_initialization(self.advantage_linear.weight)
        self._weight_initialization(self.advatage_out.weight)        
        
    def forward(self, x):       
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        v = F.relu(self.value_linear(x))
        v = self.value_out(v)
        a = F.relu(self.advantage_linear(x))
        a = self.advatage_out(a)
        q = v + a - a.mean()
        return q
