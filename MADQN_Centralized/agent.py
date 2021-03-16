import numpy as np
import torch
import os
import random
from torch.autograd import Variable
from madqn import MADQN

class Agent:
    def __init__(self, args):
        self.args = args
        self.model = MADQN(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           

    def select_action(self, o, epsilon):                
        # epsilon annealing 추가 해야함
        if np.random.uniform() < epsilon:
            u = np.random.randint(0, self.args.action_shape)
                        
        else:
            inputs = Variable(torch.from_numpy(o)).type(torch.cuda.FloatTensor)
            u = self.model.policy_network(inputs.unsqueeze(0))[0].data.cpu().numpy()                                                
            u = np.argmax(u)
        return u

    def learn(self, buffer, time_step):
        loss = self.model.train(buffer, time_step)
        return loss

        