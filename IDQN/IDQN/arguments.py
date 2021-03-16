import argparse

"""
Here are the param for the training

"""
def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for drone search")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_scenario", help="name of the scenario")
    parser.add_argument("--map-size", type=int, default=40, help="map size")
    parser.add_argument("--num-drones", type=int, default = 5, help="number of drones")
    parser.add_argument("--view-range", type=int, default = 5, help="view range")
    parser.add_argument("--num-trees", type=int, default = 0, help="number of trees")
    parser.add_argument("--num-humans", type=int, default= 20, help="number of humans")    
    # Define movement of the agents
    parser.add_argument("--action-shape", type=int, default=5, help="number of possible actions: up, down, right, and left")
    # Episode lengths and time steps    
    parser.add_argument("--max-episode-len", type=int, default=10, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")    
    # Core training parameters
    parser.add_argument("--lr-policy", type=float, default=0.00025, help="learning rate of policy")        
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon greedy")    
    parser.add_argument("--epsilon-end", type=float, default=0.15, help="epsilon end value")    
    parser.add_argument("--anneal-range", type=int, default=100000, help="annealing rate")    
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")    
    parser.add_argument("--update-target-frequency", type=int, default=3000, help="number of episodes to optimize at the same time")
    parser.add_argument("--wait-steps", type=int, default=1000, help="number of samples to collect before start training")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=1, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=50, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=100, help="how often to evaluate model")
    parser.add_argument("--tensorboard-dir", type=str, default = "./logs", help = "write the model directory")
    args = parser.parse_args()

    return args
