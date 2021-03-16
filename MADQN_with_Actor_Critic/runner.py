from tqdm import tqdm
from agent import Agent
from matplotlib.gridspec import GridSpec
from torch.utils.tensorboard import SummaryWriter

import random
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, args, env): # 수정할 사항 없음. 단순 hyperparameter setting
        self.args = args
        self.epsilon = args.epsilon
        self.epsilon_end = args.epsilon_end
        self.anneal_range = args.anneal_range
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()        
        self.buffer = None         
        self.writer = SummaryWriter(log_dir='logs/test_4')
        self.print_enough_experiences = False             
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")           

    def _init_agents(self):
        agents = []
        for i in range(self.args.num_drones):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents
    
    def rescale_actions(self, policy_output):
        action_range = self.high_action - self.low_action
        return policy_output * action_range / 2 + (self.low_action + (0.5) * action_range)
    
    def run(self):
        returns = []
        actor_loss_matrix = []
        critic_loss_matrix = []
                
        for time_step in tqdm(range(self.args.time_steps)):                                                        

            # reset the environment
            if time_step % self.episode_limit == 0:                                
                self.env.rand_reset_drone_pos()                 
            
            # get local observations
            s = []         
            for k in range(self.args.num_drones):
                # get image                    
                sensor_image = self.env.get_drone_obs(self.env.drone_list[k])        
                sensor_image = np.reshape(sensor_image, [-1])      
                # get other agent position and index information
                [clo_info, _] = self.env.communication(k)
                s_temp = np.hstack((sensor_image, clo_info))                                                                     
                # append information into a state
                s.append(s_temp)                                                           

            u = []       
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):                                                   
                    action = agent.select_action(s[agent_id], self.epsilon) # get action
                    # u.append(action)                                          
                    u.append(action)
                    actions.append(np.argmax(action))

            # get the human actions
            human_act_list = []
            for i in range(self.args.num_humans):
                human_act_list.append(-1)                
            
            # take drone actions and human actions            
            self.env.step(human_act_list, actions)   
            
            # 1) get the next states
            s_next = [] 
            next_image_matrix = []           
            next_distance_matrix =[]
            
            for k in range(self.args.num_drones): 
                sensor_image_next = self.env.get_drone_obs(self.env.drone_list[k])                
                next_image_matrix.append(sensor_image_next)
                sensor_image_next_temp = np.reshape(sensor_image_next,[-1])
                [clo_info_next, next_distance] = self.env.communication(k)
                next_distance_matrix.append(next_distance)
                s_next_temp = np.hstack((sensor_image_next_temp, clo_info_next))                         
                s_next.append(s_next_temp)      

            # get rewards
            r = self.env.get_reward(next_image_matrix, next_distance_matrix)                        
            if time_step > self.args.wait_steps:
                rewards += sum(r)
            else:
                rewards = 0                        

            image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4            
            action_length = self.args.action_shape
            reward_length = 1
            data = np.zeros([1 ,self.args.num_drones, 2 * image_length + action_length + reward_length])                        

            # [empty_size, number of agents, size of current and next state , agent and closest agent position (4*2), action (1), reward (1)]            
            for k in range(self.args.num_drones):
                # add to experience set                                
                data[:,k, 0:image_length] = s[k] # append current state                                
                data[:,k, image_length: image_length + action_length] = u[k]
                data[:,k, image_length + action_length] = r[k]
                data[:,k, image_length + action_length + reward_length:] = s_next[k]                        

            if self.buffer is None:
                self.buffer = data
            else:                
                self.buffer = np.vstack((self.buffer, data))
            
            if self.buffer is None or self.buffer.shape[0] < self.args.wait_steps or \
                self.buffer.shape[0] < self.args.batch_size:
                continue
            elif not self.print_enough_experiences:
                print('[MADQN_WITH_CRITIC] ----- generated enough experiences')
                self.print_enough_experiences = True

            if len(self.buffer) >= self.args.wait_steps:
                random_index = np.random.permutation(len(self.buffer))
                random_index = random_index[:self.args.batch_size]
                
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    [actor_loss, critic_loss] = agent.learn(self.buffer[random_index], time_step, other_agents)                    

                if time_step > self.args.wait_steps and time_step % self.episode_limit == 0:
                    actor_loss_matrix.append(actor_loss)
                    critic_loss_matrix.append(critic_loss)

            # drop from memory if too many elements
            if self.buffer.shape[0] > self.args.buffer_size:
                self.buffer = self.buffer[self.buffer.shape[0] - self.args.buffer_size:, :, :]                                 
                            
            if time_step > self.args.wait_steps and time_step % self.episode_limit == 0:
                self.writer.add_scalar('reward/episode_number', rewards, int(time_step/self.episode_limit))
     
                rewards = 0
            
            # annealing exploration for initial sampling
            if self.epsilon >= self.epsilon_end:
                self.epsilon += -(self.epsilon - self.epsilon_end)/self.anneal_range
            elif self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end            
            np.save(self.save_path + '/returns.pkl', returns)   

            # evaluate episodes
            if time_step > self.args.wait_steps and time_step % self.args.evaluate_rate == 0:
                _ = self.evaluate(time_step)                                  

    def evaluate(self, train_step):
        returns = []
        rewards = 0
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            self.env.rand_reset_drone_pos() 
            
            # visualization
            fig = plt.figure()
            gs = GridSpec(1,2, figure = fig)
            ax1 = fig.add_subplot(gs[0:1, 0:1]) # shows the true full observation
            ax2 = fig.add_subplot(gs[0:1, 1:2]) # shows the joint observation        

            for time_step in range(self.args.evaluate_episode_len):      
                # visualization                 
                ax1.imshow(self.env.get_full_obs())
                ax2.imshow(self.env.get_joint_obs())            
                plt.pause(0.01)
                plt.draw()                                
                
                s = []            
                for k in range(self.args.num_drones):
                    # get image from local sensors                
                    sensor_image = self.env.get_drone_obs(self.env.drone_list[k])                 
                    sensor_image = np.reshape(sensor_image, [-1])      
                    # get other agent position and index information
                    [clo_info, _] = self.env.communication(k)
                    s_temp = np.hstack((sensor_image, clo_info))     
                    # append information into a state
                    s.append(s_temp)                        
                actions = []

                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):                        
                        action = agent.select_action(s[agent_id], 0)                                                                                                
                        actions.append(np.argmax(action))
                        # to check the actions
                        print('orignal action_{}:{}'.format(agent_id, action))                                                                  
                        print('action_{}:{}'.format(agent_id, np.argmax(action)))

                human_act_list = [] # get the human actions
                for i in range(self.args.num_humans):
                    human_act_list.append(-1)                          
                
                # take drone actions and human actions            
                self.env.step(human_act_list, actions)                   
                
                s_next = [] # get the next state
                next_image_matrix = []
                next_distance_matrix = []
                
                for k in range(self.args.num_drones):                    
                    sensor_image_next = self.env.get_drone_obs(self.env.drone_list[k])
                    sensor_image_next_temp = np.reshape(sensor_image_next,[-1])
                    next_image_matrix.append(sensor_image_next)
                    [clo_info_next, min_dis] = self.env.communication(k)
                    next_distance_matrix.append(min_dis)
                    s_next_temp = np.hstack((sensor_image_next_temp, clo_info_next))         
                    s_next.append(s_next_temp)                         

                # get rewards
                r = self.env.get_reward(next_image_matrix, next_distance_matrix)   
                rewards += sum(r)
                # print('a1:{}, a2:{}, a3:{}, a4:{}'.format(r[0], r[1], r[2], r[3]))
            returns.append(rewards)                            
            plt.close()      
            print('Returns is:{}'.format(rewards))
            self.writer.add_scalar('reward/episode_number', rewards, int(train_step/self.episode_limit))

        return sum(returns) / self.args.evaluate_episodes


     
