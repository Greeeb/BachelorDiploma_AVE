import gymnasium
import highway_env  # required to build custom highway environment
from highway_env.envs.common.observation import OccupancyGridObservation 
from highway_env.envs.common.action import ContinuousAction

from stable_baselines3 import SAC

import random
import numpy as np
from matplotlib import pyplot as plt

iterations = int(50000)

def setup_environment():
    """
    This function sets up the environment also adjusting 
    the given configuration params to the given values
    """
    env = gymnasium.make("highway-fast-v0", 
                         render_mode="rgb_array", 
                         config={
                             "action": {
                                 "type": "ContinuousAction" # SAC works only with continuous actions
                             },
                             "observation": {
                                "type": "Kinematics",
                                "vehicles_count": 15,
                                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                                "features_range": {
                                    "x": [-100, 100],
                                    "y": [-100, 100],
                                    "vx": [-20, 20],
                                    "vy": [-20, 20]
                                },
                                "absolute": False,
                                "order": "sorted"
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

def check_model(env, model):
    for episode in range(100):
        (obs, info), done, truncated = env.reset(), False, False
        while not (done or truncated):
            print(obs)
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, truncated, info = env.step(action)
            
            env.render()
        print("done - ", done, "truncated - ", truncated)

def main():
    # Setup the environment
    env = setup_environment()
    env.reset()
    obs_class = OccupancyGridObservation(env=env)
    act_class = ContinuousAction(env=env)

    # There are 3 policies: MlpPolicy, CnnPolicy, MultiInputPolicy
    model = SAC("MlpPolicy", env, 
                verbose=1, 
                tensorboard_log="highway_sac/",
                learning_rate=5e-4,
                batch_size=32,
                buffer_size=15000,
                gamma=0.8,
                 )
    # model.learn(total_timesteps=iterations, log_interval=4, progress_bar=True)
    # model.save(f"model_sac_{iterations}")

    # del model # remove to demonstrate saving and loading
    model = SAC.load(f"model_sac_{iterations}")

    check_model(env, model)


    # # Matplotlib captures the last frame if it is needed
    # plt.imshow(env.render())
    # plt.show()


if __name__=="__main__":
    main()