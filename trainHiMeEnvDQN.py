import gymnasium as gym
import numpy as np

import os, tqdm, time

from Functions import *
from stable_baselines3 import DQN

# TODO: Always check the iterations 
iterations = 50000

def main():

    # Setting up the models
    env = setup_merge_env()

    # Loading the model
    model_path = find_model_path(iter=iterations, last=True, copy_num=0, model_type="dqn")
    model = DQN.load(model_path, env=env)

    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=iterations, last=True, copy_num=2, model_type="dqn"))

if __name__=="__main__":
    main()