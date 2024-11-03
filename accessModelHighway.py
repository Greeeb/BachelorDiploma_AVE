import gymnasium as gym
import numpy as np

import os, tqdm, time, torch, statistics

from Functions import *
from stable_baselines3 import DQN

# TODO: Always check the iterations 
iterations = 50000
seed = 200
copy_num = 2
episodes = 100

def main():

    # Setting up the models
    env = setup_highway_env()

    # Loading the model
    model_path = find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn")
    model = DQN.load(model_path)

    # Initialise results class
    results = Results()

    for episode in tqdm.tqdm(range(episodes)):
        # Resetting values for every episode 
        (obs, info), done, truncated = env.reset(seed=seed+episode), False, False
        episode_rewards = {
            'general_reward': 0,
            'collision_reward': 0,
            'right_lane_reward': 0,
            'high_speed_reward': 0,
            'on_road_reward': 0}
        criticality = []
        

        start_time = time.time()    
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            q_value = model.q_net(torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)).tolist()[0]
            criticality.append(statistics.variance(q_value))  # finding the variance of the list of q values to get criticality

            # Keep track of all the rewards for the episode summing them
            episode_rewards["general_reward"] += reward
            for key in info["rewards"].keys():
                episode_rewards[key] += info["rewards"][key]

            # Show the rendered environment
            render_last_state = env.render()
            
        # Capture the time of the epoisode
        episode_time = time.time() - start_time

        # Taking a record of all the last states for every episode
        results.append([np.array(render_last_state), episode_rewards, done, truncated, episode_time, np.array(criticality)])

    # Save results
    results.save(copy_num=copy_num)


if __name__=="__main__":
    main()