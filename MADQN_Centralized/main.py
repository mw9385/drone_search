from runner import Runner
from arguments import get_args
from env_Drones import EnvDrones
import numpy as np
import random
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()

    # define the observation size
    args.obs_shape = args.view_range * 2 - 1
    # define the environemnt
    map_size = args.map_size
    num_drones = args.num_drones
    view_range = args.view_range
    num_trees = args.num_trees
    num_humans = args.num_humans
    env = EnvDrones(map_size, num_drones, view_range, num_trees, num_humans)
        
    runner = Runner(args, env)    
    
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
