import gymnasium
import highway_env  # required to build custom highway environment
import random
from matplotlib import pyplot as plt
from stable_baselines3 import SAC

# Parameters to change in the environment configuration
CONFIG_CHANGE = [
    ("screen_width", 3000), 
    ("screen_height", 750),
    ("scaling", 25)]


def setup_environment(config_change):
    """
    This function sets up the environment also adjusting 
    the given configuration params to the given values
    """
    env = gymnasium.make("highway-v0", 
                         render_mode="rgb_array", 
                         config={
                             "action": {
                                 "type": "ContinuousAction" # "DiscreteMetaAction" # 
                             }
                         })
    env.reset()
    for change in config_change:
        env.unwrapped.config[change[0]] = change[1]
    
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
    env = setup_environment(config_change=CONFIG_CHANGE)
    env.reset()

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100, log_interval=10)
    model.save("sac_pendulum")

    del model # remove to demonstrate saving and loading

    model = SAC.load("sac_pendulum")


    obs, reward, done, trucated, info = ([], 0, False, False, {})
    for _ in range(1000):
        # action = env.unwrapped.action_type.actions_indexes["IDLE"]
        # action = find_action(obs, reward, done, trucated, info)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trucated, info = env.step(action=action)
        
        print(obs)
        
        # Stopping the environment if collision occured 
        # or the object is out of bounds/time limit is reached
        if done or trucated:
            bobs, info = env.reset()

        env.render()


    # # Matplotlib captures the last frame if it is needed
    # plt.imshow(env.render())
    # plt.show()


if __name__=="__main__":
    main()