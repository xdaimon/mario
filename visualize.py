import pygame
import pandas as pd
import numpy as np
import os, pickle, sys, time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as POSSIBLE_MOVES
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Visualize on the CPU so we can visualize and train at the same time
os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'
import tensorflow as tf
from TrainingInfo import TrainingInfo

# TODO could get num steps taken by agent by looking at the agent_stats and trainingInfo

trainingInfo = TrainingInfo('data/trainingInfo.pkl')
print(trainingInfo)

env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'), POSSIBLE_MOVES)
observation = env.reset().astype(np.float32)

pygame.init()
display = pygame.display.set_mode((2*1024,1024))

# for screen recording with OBS studio
# from time import sleep
# sleep(15)

reward_sum = 0
env_steps = 0
running = True

from Model import Parameters, Model
params = Parameters(filename='data/weights.max.npz')
model = Model(1, observation, params, visualize=True)
env_id = 0

while env_steps < trainingInfo.max_num_steps and running:
    # time.sleep(.016)

    action, activations = model(observation, env_id)
    observation, reward, done, info = env.step(action)
    observation = observation.astype(np.float32)
    reward_sum += reward
    sys.stdout.write("\033[2K\033[1G\r")
    sys.stdout.flush()
    sys.stdout.write("reward: " + str(reward_sum))
    sys.stdout.flush()
    dead = reward < -14
    if done or dead:
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if running:
        activations = [a.numpy() for a in activations]
        for i,activation in enumerate(activations):
            activation = np.swapaxes(activation,0,1).astype('uint8')
            display.blit(pygame.surfarray.make_surface(activation), (256*(i%8),256*(i//8)))
    pygame.display.update()
    env_steps += 1

pygame.quit()
print()
