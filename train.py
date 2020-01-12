import numpy as np
import time, sys, os, threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # make tf less noisy
import tensorflow as tf
from Worker import Workers
from DataSaver import DataSaver
from TrainingInfo import TrainingInfo
from Model import Model, Parameters

LOAD_WEIGHTS = False
SAVE_WEIGHTS = True
SAVE_STATS = True
LOAD_WEIGHTS_FILENAME = 'weights.mostrecent.npz'
DATA_DIR = 'data/'

sigma = .07
alpha = .01
max_steps = 50
population_size = 32
number_workers = 4
envs_per_worker = population_size // number_workers
assert(population_size % number_workers == 0)
assert(population_size % 2 == 0)

workers = Workers(number_workers, envs_per_worker)
observations = workers.get_first_observations()

dataSaver = DataSaver(DATA_DIR)
thread = None

if LOAD_WEIGHTS:
    trainingInfo = TrainingInfo(DATA_DIR+'trainingInfo.pkl')
    params = Parameters(population_size, sigma, alpha, DATA_DIR+LOAD_WEIGHTS_FILENAME)
else:
    trainingInfo = TrainingInfo()
    params = Parameters(population_size, sigma, alpha)

model = Model(population_size, observations[0], params)
rewards = np.zeros(population_size, dtype=np.float32)

max_steps = max(50, trainingInfo.max_num_steps)
number_param_updates = trainingInfo.num_episodes
mean_episode_time = None
stopping_times = {}
top3_agents_avg_stopping_time = 0
avg_stopping_time = 0
prev_steps_per_second = 150

while True:
    t0 = time.time()
    number_steps = 0
    while True:

        # launch workers
        t1 = time.time()
        for worker_id, active_envs in workers.items():
            if len(active_envs) == 0:
                continue
            envs_actions = []
            for env_id in active_envs:
                action = model(observations[env_id], env_id)
                envs_actions.append((env_id, action))
            workers.step_envs_once(worker_id, envs_actions, max_steps)

        # join workers
        number_active_envs = 0
        for results in workers.results():
            for (worker_id, env_id, observation, reward, done, info, action) in results:
                observations[env_id] = observation
                rewards[env_id] += reward
                stopping_times[env_id] = number_steps

                if done:
                    workers.set_env_done(worker_id, env_id)
                else:
                    number_active_envs += 1

                if SAVE_STATS:
                    dataSaver.append_row(info, number_param_updates, number_steps, worker_id, env_id, reward, action)

        if number_active_envs == 0:
            break

        t2 = time.time()

        if mean_episode_time is None:
            mean_episode_time = t2 - t0
        else:
            mean_episode_time = mean_episode_time*.9+.1*(t2-t0)

        prev_steps_per_second = prev_steps_per_second*.95+.05*(number_active_envs*(.5+1/(t2-t1)))

        sys.stdout.write("\033[2K\033[1G\r")
        sys.stdout.flush()
        sys.stdout.write("GradSteps {}    EnvStep {}/{}    Alive {}    StepsPerSec {}    TopMean {}    TopMax {}    TopMin {}   AvgStoppingTimeOfTop3 {}   AvgStoppingTime {}".format(number_param_updates, number_steps, max_steps, number_active_envs, int(prev_steps_per_second), int(trainingInfo.top_5_means[-1]), int(trainingInfo.top_5_maxs[-1]), int(trainingInfo.top_5_mins[-1]), top3_agents_avg_stopping_time, avg_stopping_time))
        sys.stdout.flush()
        number_steps += 1

    # Compute next max_steps
    avg_stopping_time = int(sum(stopping_times.values())/len(stopping_times))
    top3_agents_stopping_times = [stopping_times[j] for j in rewards.argsort()[-3:]]
    top3_agents_avg_stopping_time = int(sum(top3_agents_stopping_times)/3 + .5)
    max_steps = int(max(50, max_steps, 1.333*min(avg_stopping_time, top3_agents_avg_stopping_time)))

    # #max_steps = max(max_steps,int(1.25*np.sum(np.sort(rewards)[-2:])/2+.5))
    # max_steps = max(max_steps,int(1.15*np.max(rewards)+.5))

    trainingInfo.max_num_steps = max_steps

    model.reset_rnn_states()

    workers.set_envs_active()

    # Compute basic statistics
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    reward_max = np.max(rewards)
    reward_min = np.min(rewards)
    max_individual_index = np.argmax(rewards)
    min_individual_index = np.argmin(rewards)

    if SAVE_WEIGHTS:
        mean_params = [p.mean.numpy() for p in params.all()]
        max_params = [p.population[max_individual_index].numpy() for p in params.all()]
        min_params = [p.population[min_individual_index].numpy() for p in params.all()]

        top5means = trainingInfo.top_5_means
        top5maxs = trainingInfo.top_5_maxs
        top5mins = trainingInfo.top_5_mins

        # update most recent weights
        trainingInfo.num_episodes = number_param_updates
        np.savez_compressed(DATA_DIR+'weights.mostrecent.npz', *mean_params)

        # record best stats
        if reward_mean > min(top5means):
            if reward_mean > max(top5means):
                np.savez_compressed(DATA_DIR+'weights.mean.npz', *mean_params)
                trainingInfo.best_mean_episode = number_param_updates
            top5means.pop(0)
            top5means.append(reward_mean)
            top5means.sort()
        if reward_max > min(top5maxs):
            if reward_max > max(top5maxs):
                np.savez_compressed(DATA_DIR+'weights.max_mean.npz', *mean_params)
                np.savez_compressed(DATA_DIR+'weights.max.npz', *max_params)
                trainingInfo.best_max_episode = number_param_updates
            top5maxs.pop(0)
            top5maxs.append(reward_max)
            top5maxs.sort()
        if reward_min > min(top5mins):
            if reward_min > max(top5mins):
                np.savez_compressed(DATA_DIR+'weights.min.npz', *min_params)
                trainingInfo.best_min_episode = number_param_updates
            top5mins.pop(0)
            top5mins.append(reward_min)
            top5mins.sort()

        # record if mean,max,min were all in their top 5s
        # (a heuristic mean_params being a particularly good set params)
        if reward_mean in top5means and reward_max in top5maxs and reward_min in top5mins:
            np.savez_compressed(DATA_DIR+'weights.alltop5.npz', *mean_params)
            trainingInfo.hit_all_top_5_episode = number_param_updates

        trainingInfo.save(DATA_DIR+'trainingInfo.pkl')

    if SAVE_STATS:
        if thread is not None and thread.is_alive():
            thread.join()
            thread = threading.Thread(target=dataSaver.save, args=([True])).start()
        else:
            thread = threading.Thread(target=dataSaver.save, args=([False])).start()

    params.update(rewards, reward_mean, reward_std)
    rewards *= 0.
    number_param_updates += 1
    print()
