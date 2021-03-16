import numpy as np
import torch
import os
import random
from maddpg import MADDPG
from torch.autograd import Variable
from utils import gumbel_softmax
class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           

    def select_action(self, o, epsilon):                
        if np.random.uniform() < epsilon:
            # u = np.random.randint(0, 4)
            u = np.random.rand(self.args.action_shape)
            u = self.softmax(u)

        else:
            inputs = Variable(torch.from_numpy(o)).type(torch.cuda.FloatTensor)
            u = self.policy.actor_network(inputs.unsqueeze(0))[0].data.cpu().numpy()                
        return u

    def learn(self, transitions, time_step, other_agents):
        [actor_loss, critic_loss] = self.policy.train(transitions, time_step, other_agents)
        return actor_loss, critic_loss

    def softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        
        return y


