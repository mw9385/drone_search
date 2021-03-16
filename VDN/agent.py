import numpy as np
import torch
import os
import random
from policy import Policy
from trainer import Trainer
from torch.autograd import Variable

class Agent:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           
        self.model = Policy(args, agent_id).to(self.device)
        
    def select_action(self, o, epsilon):                
        inputs = Variable(torch.from_numpy(o)).type(torch.cuda.FloatTensor)
        
        with torch.no_grad():
            # epsilon annealing 추가 해야함
            if np.random.uniform() < epsilon:
                q_values = self.model(inputs.unsqueeze(0))[0].detach().cpu().numpy()                                                
                # q_values = self.model.policy_network(inputs.unsqueeze(0))[0].data.cpu().numpy()
                u = np.random.uniform(0, 1, self.args.action_shape)       
                action = np.argmax(u)     
                q_value = q_values[action]

            else:            
                q_values = self.model(inputs.unsqueeze(0))[0].detach().cpu().numpy()    
                u = q_values
                action = np.argmax(u)
                q_value = q_values[action]            

        return q_value, action        
