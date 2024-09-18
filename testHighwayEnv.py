import gymnasium
import highway_env  # required to build custom highway environment
from highway_env.envs.common.observation import OccupancyGridObservation 
from highway_env.envs.common.action import ContinuousAction

from stable_baselines3 import SAC

import random
import numpy as np
from matplotlib import pyplot as plt


def setup_environment():
    """
    This function sets up the environment also adjusting 
    the given configuration params to the given values
    """
    env = gymnasium.make("highway-v0", 
                         render_mode="rgb_array", 
                         config={
                             "action": {
                                 "type": "ContinuousAction" # "DiscreteMetaAction" 
                             },
                             "observation": {
                                 "type": "OccupancyGrid",
                                 "vehicles_count": 15,
                                 "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                                 "features_range": {
                                     "x": [-100, 100],
                                     "y": [-100, 100],
                                     "vx": [-20, 20],
                                     "vy": [-20, 20]
                                 },
                                 "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                                 "grid_step": [5, 5],
                                 "absolute": False
                             },
                             "screen_width": 2500, 
                             "screen_height": 750,
                             "scaling": 25
                         }
                     )
    env.reset()

    return env


def find_action(obs, reward, done, trucated, info):
    """
    Functions chooses one action fromthe dictionary
    Dict of all available actions:
    ACTIONS_ALL = {
            0: 'LANE_LEFT',
            1: 'IDLE',
            2: 'LANE_RIGHT',
            3: 'FASTER',
            4: 'SLOWER'
    }
    returns index
    """
    return random.randint(0, 4)


def main():
    # Setup the environment
    env = setup_environment()
    env.reset()
    obs_class = OccupancyGridObservation(env=env)
    act_class = ContinuousAction(env=env)

    # There are 3 policies: MlpPolicy, CnnPolicy, MultiInputPolicy
    model = SAC("MlpPolicy", env, verbose=2)
    model.learn(total_timesteps=1000, log_interval=4, progress_bar=True, tb_log_name="SAC")
    model.save("sac_highway")

    # del model # remove to demonstrate saving and loading
    # model = SAC.load("sac_highway")

    # obs, reward, done, trucated, info = (np.zeros((5,5)), 0, False, False, {})
    obs, info = env.reset()
    for _ in range(1000):
        # action = env.unwrapped.action_type.actions_indexes["IDLE"]
        # action = find_action(obs, reward, done, trucated, info)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trucated, info = env.step(action=action)
        
        print(obs)
        
        # Stopping the environment if collision occured 
        # or the object is out of bounds/time limit is reached
        if done or trucated:
            obs, info = env.reset()

        env.render()


    # # Matplotlib captures the last frame if it is needed
    # plt.imshow(env.render())
    # plt.show()


if __name__=="__main__":
    main()