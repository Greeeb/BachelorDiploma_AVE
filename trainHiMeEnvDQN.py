import gymnasium as gym
import numpy as np

import os, tqdm, time

from Functions import *
from stable_baselines3 import DQN

# TODO: Always check the iterations 
iterations = 20000

def main():

    # Setting up the models
    env = setup_merge_env()

    # Loading the model
    model_path = find_model_path(iter=100000, last=True, copy_num=0, model_type="dqn")
    model = DQN('MlpPolicy', env=env, exploration_fraction=0.7, seed=100, # make sure to keep seed same
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=500,
                device="cuda:1"
    ).load(model_path, env=env)

    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=100000, last=True, copy_num=3, model_type="dqn"))

if __name__=="__main__":
    main()