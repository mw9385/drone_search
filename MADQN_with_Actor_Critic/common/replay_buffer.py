import threading
import numpy as np


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()   
    
        for i in range(self.args.num_drones):            
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape * self.args.obs_shape *3 + 2 + 2]) # 이건 나중에 cnn 쓸 때 바꿔서 사용하자            
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape])
            self.buffer['r_%d' % i] = np.empty([self.size])        
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape * self.args.obs_shape *3  + 2 + 2])               
            self.buffer['done_%d' % i] = np.empty([self.size])   
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):        
        idxs = self._get_storage_idx(inc=1)        

        for i in range(self.args.num_drones):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]                
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}        
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
