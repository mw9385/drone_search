from tqdm import tqdm
from agent import Agent
from matplotlib.gridspec import GridSpec
from madqn import MADQN
from torch.utils.tensorboard import SummaryWriter

import random
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.epsilon_end = args.epsilon_end
        self.anneal_range = args.anneal_range
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agent = Agent(args)
        # self.agents = self._init_agents()        
        self.buffer = None
        self.writer = SummaryWriter(log_dir=self.args.tensorboard_dir)
        self.print_enough_experiences = False
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")               
    
    def run(self):        
        
        returns = []        
        loss_matrix = []        
        """
        변수 설명:
        s = states, o = observation, s_next = next state, r = reward,
        image_matrix = sensor image를 임시로 담기위해 만든 변수
        clo_index = closest_agent_index, clo_pos = closest_agent_position        
        """

        for time_step in tqdm(range(self.args.time_steps)):                                
                        
            if time_step % self.episode_limit == 0:
                print("Episode Number:{}:".format(int(time_step/self.episode_limit)))
                
                
                # reset the environment
                self.env.rand_reset_drone_pos() 
                
                # get local observations
            s = []                 
            for k in range(self.args.num_drones):
                # get image
                sensor_image = self.env.get_drone_obs(self.env.drone_list[k])                     
                sensor_image = np.reshape(sensor_image, [-1])      
                # get other agent position and index information
                [clo_info,  min_distance] = self.env.communication(k)
                s_temp = np.hstack((sensor_image, clo_info))
                # append information into a state
                s.append(s_temp)                        

            u = []                         
            with torch.no_grad():
                for agent_id in range(self.args.num_drones):
                    action = self.agent.select_action(s[agent_id], self.epsilon)
                    u.append(action)                                    

            # get the human actions
            human_act_list = []
            for i in range(self.args.num_humans):                
                # human_act_list.append(np.random.randint(5))  
                human_act_list.append([-1])
            
            # take drone actions and human actions            
            self.env.step(human_act_list, u)                           

            # 1) get the next states
            s_next = []            
            next_image_matrix = []
            next_distance_matrix =[]

            for k in range(self.args.num_drones): 
                # local observation for actor                
                sensor_image_next = self.env.get_drone_obs(self.env.drone_list[k])
                sensor_image_next_temp = np.reshape(sensor_image_next,[-1])
                next_image_matrix.append(sensor_image_next)
                [clo_info_next, min_dis] = self.env.communication(k)
                next_distance_matrix.append(min_dis)
                s_next_temp = np.hstack((sensor_image_next_temp, clo_info_next))         
                s_next.append(s_next_temp)  
            
            # get rewards
            r = self.env.get_reward(next_image_matrix, next_distance_matrix)
            if time_step > self.args.wait_steps:
                rewards += sum(r)
            else:
                rewards = 0

            image_length = self.args.obs_shape * self.args.obs_shape * 3 + 4
            data = np.zeros([self.args.num_drones, 2 * image_length + 1 + 1])
            # [size of current and next state , agent and closest agent position (4*2), action (1), reward (1)]
            
            for k in range(self.args.num_drones):
                # add to experience set                                
                data[k, 0:image_length] = s[k] # append current state
                data[k, image_length] = u[k]
                data[k, image_length + 1] = r[k]
                data[k, image_length +2:] = s_next[k]                        

            if self.buffer is None:
                self.buffer = data
            else:
                self.buffer = np.vstack((self.buffer, data))

            if self.buffer is None or self.buffer.shape[0] < self.args.wait_steps or \
                self.buffer.shape[0] < self.args.batch_size:
                continue
            elif not self.print_enough_experiences:
                print('[MADQN] ----- generated enough experiences')
                self.print_enough_experiences = True
            
            # update
            if len(self.buffer) >= self.args.wait_steps:
                loss = self.agent.learn(self.buffer, time_step)
                if time_step > self.args.wait_steps and time_step % self.episode_limit == 0:
                    loss_matrix.append(loss)
            
            # drop from memory if too many elements
            if self.buffer.shape[0] > self.args.buffer_size:
                self.buffer = self.buffer[self.buffer.shape[0] - self.args.buffer_size, :]

            # annealing exploration for initial sampling
            if self.epsilon >= self.epsilon_end:
                self.epsilon += -(self.epsilon - self.epsilon_end)/self.anneal_range
            else:
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
            s = []            

            self.env.rand_reset_drone_pos() 
            for k in range(self.args.num_drones):
                # get image from local sensors
                sensor_image = self.env.get_drone_obs(self.env.drone_list[k])     
                sensor_image = np.reshape(sensor_image, [-1])      
                # get other agent position and index information
                [clo_info, _] = self.env.communication(k)
                s_temp = np.hstack((sensor_image, clo_info))                     
                # append information into a state
                s.append(s_temp)        

            # visualization
            fig = plt.figure()
            gs = GridSpec(1,2, figure = fig)
            ax1 = fig.add_subplot(gs[0:1, 0:1]) # shows the true full observation
            ax2 = fig.add_subplot(gs[0:1, 1:2]) # shows the joint observation

            for time_step in range(self.args.evaluate_episode_len):      
                # visualization                 
                ax1.imshow(self.env.get_full_obs())
                ax2.imshow(self.env.get_joint_obs())
                plt.title('MADQN_Centralized')
                plt.pause(0.01)
                plt.draw()                                
                
                actions = []
                
                for agent_id in range(self.args.num_drones):
                    action = self.agent.select_action(s[agent_id],0)
                    actions.append(action)
                    # to check the actions
                    # print('action_{}:{}'.format(agent_id, action))                                            
                                        
                human_act_list = [] # get the human actions
                for i in range(self.args.num_humans):
                    # human_act_list.append(random.randint(0, 5))                                
                    human_act_list.append([-1])

                # take drone actions and human actions            
                self.env.step(human_act_list, actions)                   
                                
                # get the next state
                s_next = [] 
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
                s = s_next
            returns.append(rewards)
            print('Returns is:{}'.format(rewards))
            self.writer.add_scalar('reward/episode_number', rewards, int(train_step/self.episode_limit))              
            plt.close()
        return sum(returns) / self.args.evaluate_episodes


     