import pickle

class TrainingInfo:
    def __init__(self, filepath=''):
        if filepath:
            with open(filepath,'rb') as f:
                tInfo = pickle.load(f)
                for k,v in tInfo.__dict__.items():
                    self.__setattr__(k,v)
        else:
            self.num_episodes = 0
            self.best_mean_episode = 0
            self.best_max_episode = 0
            self.best_min_episode = 0
            self.top_5_means = [0,0,0,0,0]
            self.top_5_maxs = [0,0,0,0,0]
            self.top_5_mins = [0,0,0,0,0]
            self.hit_all_top_5_episode = 0
            self.max_num_steps = 0

    def save(self, filepath):
        with open(filepath,'wb') as f:
            pickle.dump(self,f)

    def __repr__(self):
        ret = '\nTraining Info\n'
        for k,v in self.__dict__.items():
            ret += '{:18} : {}\n'.format(k,v)
        return ret + '\n'
