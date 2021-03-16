import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

# 이 코드는 agent의 수와 target의 수가 고정된 상황에서 움직이는 target을 찾는 문제를 위해 만든 코드임
class Drones(object):
    def __init__(self, pos, view_range):
        self.pos = pos
        self.view_range = view_range

class Human(object):    
    #사람 위치도 random하게 설정할 수 있음. 나중에 constant velocity 모델을 고려한다고 하면 더 다양한 움직임을 가지는 target을 생성 가능함
    def __init__(self, pos):
        self.pos = pos

class EnvDrones(object):
    def __init__(self, map_size, drone_num, view_range, tree_num, human_num):
        self.map_size = map_size
        self.drone_num = drone_num
        self.tree_num = tree_num # tree가 사람의 움직임을 방해하는가?
        self.human_num = human_num
        self.view_range = view_range
        
        # initialize blocks 
        self.land_mark_map = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if random.random() < 0.0001:
                    # self.land_mark_map[i, j] = 0.0    # Block을 random하게 생성하는 코드
                    self.land_mark_map[i, j] = 1    # Block을 random하게 생성하는 코드

        # intialize tree
        for i in range(self.tree_num):
            temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
            while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
                temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
            self.land_mark_map[temp_pos[0], temp_pos[1]] = 2

        # initialize humans
        self.human_list = []
        random_pos_x = np.random.permutation(self.map_size)
        random_pos_y = np.random.permutation(self.map_size)        
        
        for i in range(self.human_num):            
            temp_pos = [random_pos_x[i], random_pos_y[i]]
            while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
                temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
            temp_human = Human(temp_pos)
            self.human_list.append(temp_human)

        """
        블록과 나무 그리고 사람을 초기화 할 때는 서로 겹치지 않도록 0이 아닌 부분을 제외하는 방식으로 위치 할당
        """
        # initialize drones
        self.start_pos = [self.map_size-1, self.map_size-1] # 드론의 초기 위치를 동일하게 할당하고 아래쪽에 random 한 위치로 할당 시켜버린다.
        self.drone_list = []
        for i in range(drone_num):
            temp_drone = Drones(self.start_pos, view_range)
            self.drone_list.append(temp_drone)            

    def get_full_obs(self):
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.land_mark_map[i, j] == 1: # block
                    obs[i, j, 0] = 0
                    obs[i, j, 1] = 0
                    obs[i, j, 2] = 0
                if self.land_mark_map[i, j] == 2: # tree
                    obs[i, j, 0] = 0
                    obs[i, j, 1] = 1
                    obs[i, j, 2] = 0

        for i in range(self.human_num):
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 0] = 1
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 1] = 0
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 2] = 0
        return obs

    def get_drone_obs(self, drone):
        obs_size = 2 * drone.view_range - 1        
        obs = np.ones((obs_size, obs_size, 3))

        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1

                for k in range(self.human_num):
                    if self.human_list[k].pos[0] == x and self.human_list[k].pos[1] == y:
                        obs[i, j, 0] = 1
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0

                if x>=0 and x<=self.map_size-1 and y>=0 and y<=self.map_size-1:
                    if self.land_mark_map[x, y] == 1:
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0
                    if self.land_mark_map[x, y] == 2:
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 1
                        obs[i, j, 2] = 0
                else:
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0.5
                    obs[i, j, 2] = 0.5

                if (drone.view_range - 1 - i)*(drone.view_range - 1 - i)+(drone.view_range - 1 - j)*(drone.view_range - 1 - j) > drone.view_range*drone.view_range:
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0.5
                    obs[i, j, 2] = 0.5
        return obs

    def get_joint_obs(self):
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                obs[i, j, 0] = 0.5
                obs[i, j, 1] = 0.5
                obs[i, j, 2] = 0.5
        for k in range(self.drone_num):
            temp = self.get_drone_obs(self.drone_list[k])
            size = temp.shape[0]            
            for i in range(size):
                for j in range(size):
                    x = i + self.drone_list[k].pos[0] - self.drone_list[k].view_range + 1
                    y = j + self.drone_list[k].pos[1] - self.drone_list[k].view_range + 1
                    if_obs = True
                    if temp[i, j, 0] == 0.5 and temp[i, j, 1] == 0.5 and temp[i, j, 2] == 0.5:
                        if_obs = False
                    if if_obs == True:                        
                        obs[x, y, 0] = temp[i, j, 0]
                        obs[x, y, 1] = temp[i, j, 1]
                        obs[x, y, 2] = temp[i, j, 2]
        return obs

    def get_local_obs(self, drone_index):
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                obs[i, j, 0] = 0.5
                obs[i, j, 1] = 0.5
                obs[i, j, 2] = 0.5
        temp = self.get_drone_obs(self.drone_list[drone_index])
        size = temp.shape[0]
        for i in range(size):
            for j in range(size):
                x = i + self.drone_list[drone_index].pos[0] - self.drone_list[drone_index].view_range + 1
                y = j + self.drone_list[drone_index].pos[1] - self.drone_list[drone_index].view_range + 1
                if_obs = True
                if temp[i, j, 0] == 0.5 and temp[i, j, 1] == 0.5 and temp[i, j, 2] == 0.5:
                    if_obs = False
                if if_obs == True:                        
                    obs[x, y, 0] = temp[i, j, 0]
                    obs[x, y, 1] = temp[i, j, 1]
                    obs[x, y, 2] = temp[i, j, 2]
        return obs
        
    def rand_reset_drone_pos(self):  
        # human_pos = [[1,1], [self.map_size-5, self.map_size - 5], [10, 20], [20, 10], [25, 25]] 
        # drone_pos = [[5,5], [9,9], [15, 15], [20, 20]]        
        for k in range(self.drone_num):                        
            self.drone_list[k].pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size -1)]                        
            # self.drone_list[k].pos = drone_pos[k]
        for k in range(self.human_num):
            self.human_list[k].pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size -1)]
            # self.human_list[k].pos = human_pos[k]

    def drone_step(self, drone_act_list):
        # Action에 맞춰서 drone의 position을 grid 상에서 옮겨버림
        if len(drone_act_list) != self.drone_num:
            print("Not enough number of actions for the agents")
            return 0
        bad_actions = []
        self.bad_action = False        
        for k in range(self.drone_num):
            if drone_act_list[k] == 0: # go up
                if self.drone_list[k].pos[0] > 0:
                    self.drone_list[k].pos[0] = self.drone_list[k].pos[0] - 1
                if self.drone_list[k].pos[0] == 0:
                    self.bad_action = True
            elif drone_act_list[k] == 1: # go down
                if self.drone_list[k].pos[0] < self.map_size - 1:
                    self.drone_list[k].pos[0] = self.drone_list[k].pos[0] + 1
                if self.drone_list[k].pos[0] == self.map_size:
                    self.bad_action = True
            elif drone_act_list[k] == 2: # go left
                if self.drone_list[k].pos[1] > 0:
                    self.drone_list[k].pos[1] = self.drone_list[k].pos[1] - 1
                if self.drone_list[k].pos[1] == 0:
                    self.bad_action = True
            elif drone_act_list[k] == 3: # go right
                if self.drone_list[k].pos[1] < self.map_size - 1:
                    self.drone_list[k].pos[1] = self.drone_list[k].pos[1] + 1
                if self.drone_list[k].pos[1] == self.map_size:
                    self.bad_action = True
            elif drone_act_list[k] == 4: # stop
                self.drone_list[k].pos[0] = self.drone_list[k].pos[0]
                self.drone_list[k].pos[1] = self.drone_list[k].pos[1]
                
                if self.drone_list[k].pos[0] == self.map_size or self.drone_list[k].pos[1] == self.map_size:
                    self.bad_action = True
            bad_actions.append(self.bad_action)

        return bad_actions
        """
        ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ> y
        |
        |
        |    x축, y축을 잘 보고 판단!
        |
        |
        |
        x
        """
    def human_step(self, human_act_list):
        if len(human_act_list) != self.human_num:
            return
        for k in range(self.human_num):
            if human_act_list[k] == 0:
                if self.human_list[k].pos[0] > 0:
                    free_space = self.land_mark_map[self.human_list[k].pos[0] - 1, self.human_list[k].pos[1]]
                    if free_space == 0:
                        self.human_list[k].pos[0] = self.human_list[k].pos[0] - 1
            elif human_act_list[k] == 1:
                if self.human_list[k].pos[0] < self.map_size - 1:
                    free_space = self.land_mark_map[self.human_list[k].pos[0] + 1, self.human_list[k].pos[1]]
                    if free_space == 0:
                        self.human_list[k].pos[0] = self.human_list[k].pos[0] + 1
            elif human_act_list[k] == 2:
                if self.human_list[k].pos[1] > 0:
                    free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] - 1]
                    if free_space == 0:
                        self.human_list[k].pos[1] = self.human_list[k].pos[1] - 1
            elif human_act_list[k] == 3:
                if self.human_list[k].pos[1] < self.map_size - 1:
                    free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] + 1]
                    if free_space == 0:
                        self.human_list[k].pos[1] = self.human_list[k].pos[1] + 1
            elif human_act_list[k] == 4:
                self.human_list[k].pos[0] = self.human_list[k].pos[0]
                self.human_list[k].pos[1] = self.human_list[k].pos[1]

    def step(self, human_act_list, drone_act_list):     
        self.drone_step(drone_act_list)
        self.human_step(human_act_list)        

    def cal_distance(self, ref_x, ref_y, target_x, target_y):
        return np.sqrt(pow(ref_x -target_x, 2) + pow(ref_y - target_y, 2))

    def get_reward(self, observations, min_distance):
        obs_size = 2 * self.view_range - 1
        communication_bound = obs_size
        total_reward = []
        distance_matrix = []
        # calculate the center position of the drone
        self.x_center = int(obs_size / 2)
        self.y_center = int(obs_size / 2)

        for k in range(self.drone_num):  
            current_x = self.drone_list[k].pos[0]
            current_y = self.drone_list[k].pos[1]
            r = -0.15
            target_count = 0                        
            temp_count = 0
            # target이 local observation 내부에 있는지 여부를 파악하여 reward를 줌
            for i in range(obs_size):
                for j in range(obs_size):
                    if observations[k][i,j,0]==1 and observations[k][i,j,1] == 0 and observations[k][i,j,2] == 0:
                        temp_count += 1
                        target_count +=1
                        
                    if temp_count != 0:                     
                        distance = self.cal_distance(self.x_center, self.y_center, i, j)                                                                                      
                        r = (self.view_range*2 - distance + 1) * 0.2
                        temp_count = 0

            # commnuication bound
            if min_distance[k] > communication_bound:
                r = r + 0.2      
            # Communication bound에 너무 큰 negative value를 주는 경우 sparse reward? problem으로 agent가 아무것도 안하는 상황이 발생함
            
            # # exploration reward
            # if target_count is 0 and np.sum(observations[k]) > self.view_range * self.view_range * 3 - 4 * 2 * 1.5:
            #     r = r + 0.2                                 

            total_reward.append(np.round(r, 3))                
            
        return total_reward

    def communication(self, agent_index):
        # get current drone position
        current_x = self.drone_list[agent_index].pos[0]
        current_y = self.drone_list[agent_index].pos[1]
        
        # calculate the distance between agents
        distance = np.zeros([self.drone_num,1])
        index = 0        
        for k in range(self.drone_num):
            if k is not agent_index:
                target_x = self.drone_list[k].pos[0]
                target_y = self.drone_list[k].pos[1]
                distance[k] = self.cal_distance(current_x, current_y, target_x, target_y)
            elif k == agent_index:
                distance[k] = 1000
        
        # get the closest agent index and position
        min_distance = np.min(distance)
        closest_agent_index = np.argmin(distance) 
        closest_position = [self.drone_list[closest_agent_index].pos[0], self.drone_list[closest_agent_index].pos[0]]       
        
        vector_pos = [current_x / self.map_size, 
                    current_y / self.map_size, 
                    self.drone_list[closest_agent_index].pos[0] / self.map_size, 
                    self.drone_list[closest_agent_index].pos[0]/ self.map_size]                                                           
        closest_info = vector_pos 
        
        return [closest_info, min_distance]
