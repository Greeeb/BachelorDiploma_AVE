import gymnasium as gym
import numpy as np

import os, tqdm, time, torch, statistics

from Functions import *
from stable_baselines3 import DQN
from scipy.signal import find_peaks  # For detecting peaks

# TODO: Always check the iterations 
iterations = 100000
seed = 200
copy_num = 0
episodes = 10

def main(copy=copy_num):
    copy_num = copy
    # Setting up the models
    env = setup_highway_env()

    # Loading the model
    model_path = find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn")
    model = DQN.load(model_path, env=env)

    for episode in tqdm.tqdm(range(episodes)):
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

            env.render()

if __name__=="__main__":
    main()