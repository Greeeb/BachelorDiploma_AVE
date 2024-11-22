from stable_baselines3 import DQN
import numpy as np
from PIL import Image
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from Functions import *


iterations = 10000
criticals_file = "model_dqn_100000(0)"  # Folder name from the criticality_renderings folder
torch.cuda.set_device(1)


def main():
    env = setup_highway_env()

    results = Results()
    results.load(
        os.path.join(os.path.abspath(os.path.curdir), "BachelorDiploma_AVE", "saveResults", criticals_file)
        )
    critical_obs = results.crit_obs
    iterations = critical_obs.shape[0]
    print(iterations)
    
    # Define the observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space
    
    # Initialize the custom replay buffer
    buffer_size = 100000
    # replay_buffer = CustomReplayBuffer(buffer_size, observation_space, action_space)

    # # Preload the replay buffer with the predefined initial states
    # replay_buffer.preload_states(critical_obs)
    
    # # Example: Load a list of states from PNG files
    # stored_states = load_state_from_png(criticals_file)

    # # Define the observation space using gymnasium.spaces.Box
    # obs_shape = (1, 84, 84)  # Channels-first format (C, H, W) for images
    # observation_space = gymnasium.spaces.Box(
    #     low=0, high=255, shape=obs_shape, dtype=np.uint8
    # )
    # # Define a discrete action space with 4 possible actions (example)
    # action_space = gymnasium.spaces.Discrete(4)
    
    # Loading the model
    model_path = find_model_path(iter=1000, last=True, copy_num=5, model_type="dqn")
    
    model = DQN('MlpPolicy', env=env, exploration_fraction=0.7, seed=100, # make sure to keep seed same
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=500,
                    replay_buffer_class=CustomReplayBuffer,
                    replay_buffer_kwargs={
                        # 'observation_space': observation_space,
                        # 'buffer_size': 1000000,
                        # 'action_space': action_space,
                    }
        ).load(model_path, env=env)
    
    # critical_obs = np.concatenate([critical_obs, critical_obs], axis=0 )
    for obs in tqdm.tqdm(critical_obs):
        action, _ = model.predict(obs, deterministic=True)
        obs_next, reward, done, truncated, info = env.step(int(action))
        model.replay_buffer.add(obs=obs, next_obs=obs_next, reward=reward, done=done, action=action, infos=info)

    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=iterations, last=True, copy_num=7, model_type="dqn"))
    
    del model
    
    # Loading the model
    model_path = find_model_path(iter=iterations, last=True, copy_num=7, model_type="dqn")
    
    model = DQN('MlpPolicy', env=env, exploration_fraction=0.7, seed=100, # make sure to keep seed same
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=500,
                    replay_buffer_class=CustomReplayBuffer,
                    replay_buffer_kwargs={
                        # 'observation_space': observation_space,
                        # 'buffer_size': 1000000,
                        # 'action_space': action_space,
                    }
        ).load(model_path, env=env)
    
    # critical_obs = np.concatenate([critical_obs, critical_obs], axis=0 )
    for obs in tqdm.tqdm(critical_obs):
        action, _ = model.predict(obs, deterministic=True)
        obs_next, reward, done, truncated, info = env.step(int(action))
        model.replay_buffer.add(obs=obs, next_obs=obs_next, reward=reward, done=done, action=action, infos=info)

    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=iterations, last=True, copy_num=8, model_type="dqn"))
    
    

if __name__=="__main__":
    main()