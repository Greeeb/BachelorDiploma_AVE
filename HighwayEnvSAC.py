import gymnasium
import highway_env  # required to build custom highway environment
from highway_env.envs.common.observation import OccupancyGridObservation 
from highway_env.envs.common.action import ContinuousAction

from rlkit.normalized_box_env import NormalizedBoxEnv


from stable_baselines3 import SAC

import random, os
import numpy as np
from matplotlib import pyplot as plt


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
            # print(obs)
            action, _ = model.predict(obs, deterministic=True)
            # print(action)
            obs, reward, done, truncated, info = env.step(action)
            
            env.render()
        print("done - ", done, "truncated - ", truncated)

def find_model_path(iter, last=False, copy_num=None):
    """
    The fucntion returns the path to the model:
    1. To the particular model if copy_num is set to int.
    2. To the last model if last set to True.
    3. To the first not existing copy number for the purpose of saving the model.

    :param iter: The learning iteratins count of the model.
    :param last: True if you need to receive path to the last saved model for given iterations number.
    :param copy_num: Set to int if you need to receive a particular copy of existing model.
    """

    copy = 0
    model_filename = f"model_sac_{iter}({copy})"
    models_path = os.path.join(os.curdir, "models")

    if copy_num != None:
        copy = copy_num
        return os.path.join(models_path, f"model_sac_{iter}({copy})")

    while f"model_sac_{iter}({copy}).zip" in os.listdir(models_path):
        copy += 1 

    if last:
        copy -= 1

    return os.path.join(models_path, f"model_sac_{iter}({copy})")

def main():
    # Setup the environment
    env = NormalizedBoxEnv(setup_environment())
    env.reset()
    # obs_class = OccupancyGridObservation(env=env)
    # act_class = ContinuousAction(env=env)

    # Setup the RL model
    # There are 3 policies: MlpPolicy, CnnPolicy, MultiInputPolicy
    model = SAC("MlpPolicy", env, 
                verbose=1, 
                tensorboard_log="logs/highway_sac/",
                learning_rate=5e-4,
                batch_size=32,
                buffer_size=15000,
                gamma=0.8,
                 )
    # Print the policy of the model
    print(model.policy)

    # Number of iterations while training
    iterations = int(20000)

    # Train the model
    model.learn(total_timesteps=iterations, log_interval=4, progress_bar=True)
    # Save the model into ./models
    model.save(find_model_path(iter=iterations))


    # # Load the existing model from ./models folder
    # # Add copy_num=int if you need to launch particular copy
    # model_path = find_model_path(iter=iterations, last=True, copy_num=None)
    # model = SAC.load(model_path)  
    # print(f"Model is loaded from: {model_path}")


    check_model(env, model)


    # # Matplotlib captures the last frame if it is needed
    # plt.imshow(env.render())
    # plt.show()


if __name__=="__main__":
    main()