from env_Drones import EnvDrones
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import numpy as np

env = EnvDrones(30, 3, 10, 30, 5)   # map_size, drone_num, view_range, tree_num, human_num
env.rand_reset_drone_pos() # drone의 시작 위치를 random하게 섞어준다. 그런데 우리는 이렇게 random한 위치에서 시작할게 아니라 동일한 위치에서 뻗어나가는 형식으로 진행해야한다.

max_MC_iter = 100
# Drone의 움직임을 보기 위해 설정
fig = plt.figure()
gs = GridSpec(1, 5, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])
ax2 = fig.add_subplot(gs[0:1, 1:2])
# See local observation
ax3 = fig.add_subplot(gs[0:5, 2:3])
ax4 = fig.add_subplot(gs[0:5, 3:4])
ax5 = fig.add_subplot(gs[0:5, 4:5])

for MC_iter in range(max_MC_iter):
    # print(MC_iter)
    ax1.imshow(env.get_full_obs())
    ax2.imshow(env.get_joint_obs())
    'env.drone_list[drone_index].pos(x,y)'
    ax3.imshow(env.get_drone_obs(env.drone_list[0]))
    ax4.imshow(env.get_drone_obs(env.drone_list[1]))
    ax5.imshow(env.get_drone_obs(env.drone_list[2]))    
    # print(np.shape(env.get_drone_obs(env.drone_list[2])))
        

    # local observation 과 global observation을 넣을때 차이:
    # global observation은 local observation을 여러장 중첩해서 넣는 방식을 사용한다.
    # global observation의 each element size는 local observation의 size와 동일해야 한다.
    # 지금 만들어진 환경에서는 local observation의 size가 global observation의 사이즈와 다르다
    # critic code를 조금 변경해서 사용할 수 도 있을것 같으니 조금더 알아보자
    s1 = env.get_drone_obs(env.drone_list[0])
    s2 = env.get_drone_obs(env.drone_list[1])
    s3 = env.get_drone_obs(env.drone_list[2])
    r1 = env.get_reward(s1)
    r2 = env.get_reward(s2)
    r3 = env.get_reward(s3)
    print('R1:{}, R2:{}, R3:{}'.format(r1,r2,r3))


    human_act_list = []
    for i in range(env.human_num):
        human_act_list.append(random.randint(0, 4))

    drone_act_list = []
    for i in range(env.drone_num):
        drone_act_list.append(random.randint(0, 4))
    env.step(human_act_list, drone_act_list)
    plt.pause(.5)
    plt.draw()

