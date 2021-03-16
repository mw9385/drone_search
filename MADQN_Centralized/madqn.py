import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from policy import Policy
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class MADQN:
    def __init__(self, args):  
        self.args = args        
        self.evaluate_rate = self.args.evaluate_rate        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.writer = SummaryWriter(log_dir=self.args.tensorboard_dir)

        # create the network
        self.policy_network = Policy(args).to(self.device)

        # build up the target network
        self.policy_target_network = Policy(args).to(self.device)

        # load the weights into the target networks
        self.policy_target_network.load_state_dict(self.policy_network.state_dict())        

        # create the optimizer
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=self.args.lr_policy)        

        # create the dict for store the model
        self.episode_limit = self.args.max_episode_len
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 저장된 모델이 있다면 불러오기 
        if os.path.exists(self.model_path + '/policy_params.pkl'):
            self.policy_network.load_state_dict(torch.load(self.model_path + '/policy_params.pkl'))            
            print('Agent successfully loaded policy_network: {}'.format(self.model_path + '/policy_params.pkl'))            

    # soft update
    def _update_target_network(self):                       
        for target_param, param in zip(self.policy_target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_((1-self.args.tau) * target_param.data + self.args.tau * param.data)
        # self.policy_target_network = copy.deepcopy(self.policy_network)                
    # update the network
    def train(self, buffer, time_step):                
        self.buffer = buffer
        image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4

        with torch.no_grad():
            # create mini batch
            batch = self.buffer[np.random.choice(self.buffer.shape[0], self.args.batch_size, replace = False), :]
            batch_state = torch.from_numpy(batch[:, 0:image_length])
            batch_state = Variable(batch_state).type(torch.cuda.FloatTensor)            
            batch_actions = torch.from_numpy(batch[:, image_length])
            batch_actions = Variable(batch_actions).type(torch.cuda.LongTensor)            
            # print(batch_actions)
            # calculate loss        
            batch_rewards = batch[:, image_length + 1]
            batch_next_states = Variable(torch.from_numpy(batch[:, image_length + 1 + 1:])).type(torch.cuda.FloatTensor)              
            tt = self.policy_target_network(batch_next_states).data.cpu().numpy()
            tt = batch_rewards + self.args.gamma*np.amax(tt, axis=1)
            tt = Variable(torch.from_numpy(tt), requires_grad = False).type(torch.cuda.FloatTensor)
        
        x = self.policy_network(batch_state).gather(1, batch_actions.view(-1,1)).squeeze()
        loss = self.loss_fn(x, tt)

        # back propagation
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        self._update_target_network()
        # if time_step % self.args.update_target_frequency == 0:
        #     self._update_target_network()                                   
        #     print('[MADQN] ---- updated target network (%d)' % (time_step/self.args.update_target_frequency))
        del batch

        if time_step % 50 == 0:
            # add to tensorboard
            self.writer.add_scalar('loss', loss, int(time_step / 50))            


        return loss

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy_network.state_dict(), model_path + '/' + num + '_policy_params.pkl')        