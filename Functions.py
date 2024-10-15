import os, gymnasium, highway_env
import numpy as np


def find_model_path(iter=50000, last=False, copy_num=None, model_type="dqn"):
    """
    The function returns the path to the model:
    1. To the particular model if copy_num is set to int.
    2. To the last model if last set to True.
    3. To the first not existing copy number for the purpose of saving the model.

    :param iter: The learning iteratins count of the model.
    :param last: True if you need to receive path to the last saved model for given iterations number.
    :param copy_num: Set to int if you need to receive a particular copy of existing model.
    :param model_type: the mypo of the model used, default="dqn"
    """

    copy = 0
    models_path = os.path.join(os.curdir, "models")

    if copy_num != None:
        copy = copy_num
        return os.path.join(models_path, f"model_{model_type}_{iter}({copy})")

    while f"model_{model_type}_{iter}({copy}).zip" in os.listdir(models_path):
        copy += 1 

    if last:
        copy -= 1

    return os.path.join(models_path, f"model_{model_type}_{iter}({copy})")


def results_path(model_path=find_model_path(iter=50000, last=True, copy_num=None, model_type="dqn")):
    """
    The function returns the path to the renders folder in saveResults:
    """
    index = 0
    general_path = os.path.join(os.curdir, "saveResults")

    folder_name = model_path.replace(str(os.path.join(os.curdir, "models")), "")
    folder_path = os.path.join(os.curdir, "saveResults" + folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("New folder for renders created:", folder_path)
    else:
        print("The folder for renders exists:", folder_path)   

    return folder_path


def setup_highway_env():
    return setup_env("highway-v0")


def setup_merge_env():
    return setup_env("merge-v0")


def setup_env(env_name):
    """
    This function sets up the environment also adjusting 
    the given configuration params to the given values
    """
    env = gymnasium.make(env_name, 
                         render_mode="rgb_array", 
                         config={
                            "screen_width": 2500, 
                            "screen_height": 750,
                            "scaling": 25
                         },
                     )
    env.reset()

    return env

class Results():
    def __init__(self):
        # Initialise all the tracking variables
        self.renders = np.zeros((1,750,2500,3))
        self.rewards, self.dones, self.truncateds, self.times = (np.array([[0],[0]]) for _ in range(4))

        # Dictionary of all variables to be tracked
        self.results_dict = {
            "renders": self.renders,
            "rewards": self.rewards, 
            "dones": self.dones, 
            "truncateds": self.truncateds,
            "times": self.times}

    def append(self, data):
        """
        Appends the data to be tracked.
        :param data: should be in order [renders, rewards, dones, truncateds, times]
        """
        self.renders = np.append(self.renders, np.array(data[0]).reshape(1,750,2500,3), axis=0)
        self.rewards = np.append(self.rewards,np.array(data[1]).reshape((1,1)), axis=0)
        self.dones = np.append(self.dones,np.array(data[2]).reshape((1,1)), axis=0)
        self.truncateds = np.append(self.truncateds,np.array(data[3]).reshape((1,1)), axis=0)
        self.times = np.append(self.times,np.array(data[4]).reshape((1,1)), axis=0)        
        
    def save(self):
        print()
        # Saving the np array of all the last states(first array is zeroes)
        for var in self.results_dict.keys():
            np.save(os.path.join(results_path(), f"{var}"), self.results_dict[var])

        
        



