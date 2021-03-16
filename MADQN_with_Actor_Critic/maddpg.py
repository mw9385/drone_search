import torch
import torch.nn.utils as torch_utils
import os
import numpy as np
import matplotlib.pyplot as plt

from actor_critic import Actor, Critic
from torch.autograd import Variable
from utils import onehot_from_logits, gumbel_softmax
from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    def __init__(self, args, agent_id):  
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0 # This value may relevant to MAX episodes
        self.evaluate_rate = self.args.evaluate_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir='logs/test_4')

        # create the network
        self.actor_network = Actor(args, agent_id).cuda()
        self.critic_network = Critic(args).cuda()

        # build up the target network
        self.actor_target_network = Actor(args, agent_id).cuda()
        self.critic_target_network = Critic(args).cuda()

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        self.episode_limit = self.args.max_episode_len
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 저장된 모델이 있다면 불러오기 
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, buffer, time_step, other_agents):        
        """
        buffer_shape = [number of samples, number of agents + o + a + r + o_next]
        """
        image_length = self.args.obs_shape * self.args.obs_shape *  3 + 4        
        action_length = self.args.action_shape
        reward_length = 1                
        torch.autograd.set_detect_anomaly(True)

        target_action = []
        state = torch.tensor(buffer[:,:, 0:image_length], dtype = torch.float32).cuda()                    
        action = torch.tensor(buffer[:, :, image_length: image_length + action_length], dtype = torch.float32).cuda()
        reward = torch.tensor(buffer[:, :, image_length + action_length], dtype = torch.float32).cuda()
        next_state = torch.tensor(buffer[:, :, image_length + action_length + reward_length:], dtype = torch.float32).cuda()                        
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.num_drones):
                temp_next = buffer[:, agent_id, image_length + action_length + reward_length:]
                temp_next = torch.tensor(np.reshape(temp_next, [self.args.batch_size, image_length]), dtype = torch.float32).cuda()

                #infer other agent's action from local polices                
                if agent_id == self.agent_id:
                    temp_next_action = self.actor_target_network(temp_next).data.cpu().numpy()
                    target_action.append(temp_next_action)  
                else:
                    temp_next_action = other_agents[index].policy.actor_target_network(temp_next).data.cpu().numpy()
                    target_action.append(temp_next_action)                
                    index +=1         

            target_action = np.transpose(target_action, (1,0,2))   
            target_action = torch.tensor(target_action, dtype = torch.float32).cuda()        

            # q next
            q_next = self.critic_target_network(next_state, target_action).detach() # shape [32, 1], 모든 state 정보를 가지고 있는 target

            # current reward
            current_reward = reward[:, self.agent_id].view(-1,1)  # shape = [32, 1], 현재 agent의 보상 함수 정보만을 가지고 있음
            # target q value
            target_q = (current_reward + self.args.gamma * q_next).detach() # shape [32, 1]

        # initialize critic optimizer
        self.critic_optim.zero_grad()        

        # define critic loss
        q_value = self.critic_network(state, action) # shape [32, 1]         
        critic_loss = (target_q - q_value).pow(2).mean()

        # update critic loss
        critic_loss.backward()
        torch_utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
        self.critic_optim.step()                   

        # initialize actor loss
        self.actor_optim.zero_grad()
        
        current_action = torch.empty((self.args.batch_size, self.args.num_drones, self.args.action_shape)).type(torch.cuda.FloatTensor)

        index = 0
        # get the current action using 'current policy'
        for agent_id in range(self.args.num_drones):
            temp_current = buffer[:, agent_id, 0:image_length]
            temp_current = torch.tensor(np.reshape(temp_current, [self.args.batch_size, image_length]), dtype = torch.float32).cuda()
            
            if agent_id == self.agent_id:
                current_action[:, agent_id, :] = self.actor_network(temp_current)                                
            else:
                current_action[:, agent_id, :] = other_agents[index].policy.actor_network(temp_current)                                
                           
        actor_loss = -self.critic_network(state, current_action).mean()
        actor_loss.backward()
        torch_utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
        self.actor_optim.step()  
        
        self._soft_update_target_network()   
        if self.train_step % 50 == 0:
            # add to tensorboard
            self.writer.add_scalar('agent_%i/actor_loss' % self.agent_id, actor_loss, int(self.train_step / 50))
            self.writer.add_scalar('agent_%i/critic_loss' % self.agent_id, critic_loss, int(self.train_step / 50))       


        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        
        self.train_step += 1
        return [actor_loss, critic_loss]

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


