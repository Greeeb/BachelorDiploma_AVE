import gymnasium as gym
import numpy as np

import os, tqdm, time, torch, statistics

from Functions import *
from stable_baselines3 import DQN
from scipy.signal import find_peaks  # For detecting peaks

# TODO: Always check the iterations 
iterations = 10000
seed = 200
copy_num = 6
episodes = 100
seed = 200
# torch.cuda.set_device(1)

def main(iterations=iterations, copy=copy_num, seed=seed, save_copy=copy_num):
    copy_num = copy
    # Setting up the models
    env = setup_highway_env()

    # Loading the model
    model_path = find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn")
    model = DQN('MlpPolicy', env=env, exploration_fraction=0.1, seed=seed, # make sure to keep seed same
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=500,
                device="cuda:1"
    ).load(model_path, env=env)
    
    # Initialise results class
    results = Results()

    for episode in tqdm.tqdm(range(episodes)):
        # Resetting values for every episode 
        (obs, info), done, truncated = env.reset(seed=seed + episode*10), False, False
        episode_rewards = {
            'general_reward': 0,
            'collision_reward': 0,
            'right_lane_reward': 0,
            'high_speed_reward': 0,
            'on_road_reward': 0
        }
        criticality = []
        all_obs = []
        all_renderings = []  # Store all renderings to filter after the episode
        timestamps = []      # Store timestamps for each rendering
        episode_start_time = time.time()

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs_next, reward, done, truncated, info = env.step(int(action))

            # Sum up rewards for the episode
            episode_rewards["general_reward"] += reward
            for key in info["rewards"].keys():
                episode_rewards[key] += info["rewards"][key]
            
            all_obs.append(obs_next)
            obs = obs_next

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        
        # Calculate criticality
        for (index, obs) in enumerate(all_obs):
            q_value = model.q_net(torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)).tolist()[0] #, device="cuda:1").flatten().unsqueeze(0)).tolist()[0]
            criticality.append(statistics.variance(q_value))
            timestamps.append(index)

        # After episode ends, find peaks in the criticality array
        peaks, _ = find_peaks(criticality)
        # peak_renderings = [(timestamps[i], all_renderings[i]) for i in peaks]  # Keep only renderings at peak indices with timestamps
        crit_obs = [all_obs[i] for i in peaks]
        
        # Append episode data to Results, including peak renderings and episode duration
        results.append([
            np.array([]), #np.array(all_renderings[-1]),  # Last state rendering
            episode_rewards, 
            done, 
            truncated, 
            episode_duration,  # Episode duration is preserved here
            np.array(criticality), 
            np.array([]),  # Ensure compatibility with mixed timestamp-rendering tuples
            crit_obs
        ])

    # Save results
    results.save(copy_num=save_copy, iter=iterations)


if __name__=="__main__":
    main()