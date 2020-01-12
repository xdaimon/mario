import os, time
import pandas as pd

class DataSaver:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.agent_stats = []
        self.count = 0
        for f in os.listdir(data_dir):
            if 'agent_stats.' in f:
                self.count += 1
        self.agent_stats_filename = 'agent_stats.'+str(self.count)+'.pkl'
        self.df = pd.DataFrame([],columns=['episode', 'worker', 'agent', 'step', 'reward', 'action', 'x_pos', 'y_pos', 'coins', 'score', 'status', 'stage'])

    def save(self, flush_to_disk):
        # measure how much time it takes to save the df
        t = time.time()
        self.df=self.df.append(self.agent_stats, ignore_index=True, sort=False).astype(int)
        self.df.to_pickle(self.data_dir+self.agent_stats_filename,compression='gzip')
        # if we took too long then we should start a new file
        if flush_to_disk:
            self.df = pd.DataFrame([],columns=['episode', 'worker', 'agent', 'step', 'reward', 'action', 'x_pos', 'y_pos', 'coins', 'score', 'status', 'stage'])
            self.count += 1
            self.agent_stats_filename = 'agent_stats.'+str(self.count)+'.pkl'
        self.agent_stats = []

    def append_row(self, info, num_param_updates, step, worker_id, env_i, reward, action):
        del info['flag_get']
        del info['life']
        del info['time']
        del info['world']
        info['status'] = {'small':0,'tall':1,'fireball':2}[info['status']]
        info['reward'] = int(reward)
        info['action'] = int(action)
        # indices
        info['agent'] = env_i
        info['worker'] = worker_id
        info['step'] = step
        info['episode'] = num_param_updates
        # end indices
        self.agent_stats.append(info)
