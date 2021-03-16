import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class Policy(nn.Module):

    def __init__(self, args):               
        super(Policy, self).__init__()
        self.args = args
        self.image_dims = self.args.obs_shape * self.args.obs_shape*3 + 4
        self.hidden_nodes = 2048

        # inputs: sensor image + closest_agent_id, closest_agnet_vector
        self.fc1 = nn.Linear(self.image_dims, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc3 = nn.Linear(self.hidden_nodes, self.args.action_shape)

    def forward(self, x):       
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        actions = F.relu(self.fc3(x))    
        # print(actions[0]) 
        return actions
