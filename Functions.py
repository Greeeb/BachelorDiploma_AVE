import os, gymnasium, highway_env, statistics, tqdm
import numpy as np
from PIL import Image

iterations = 100000

def find_model_path(iter=iterations, last=False, copy_num=None, model_type="dqn"):
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
    if os.path.abspath(os.curdir).endswith("BachelorDiploma_AVE"):
        models_path = os.path.join(os.path.abspath(os.curdir), "models")
    else:
        models_path = os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")

    if copy_num != None:
        copy = copy_num
        return os.path.join(models_path, f"model_{model_type}_{iter}({copy})")

    while f"model_{model_type}_{iter}({copy}).zip" in os.listdir(models_path):
        copy += 1 

    if last:
        copy -= 1

    return os.path.join(models_path, f"model_{model_type}_{iter}({copy})")


def results_path(model_path=find_model_path(iter=iterations, last=True, copy_num=None, model_type="dqn")):
    """
    The function returns the path to the renders folder in saveResults:
    """

    folder_path = model_path.replace("models", "saveResults")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("New folder for renders created:", folder_path)
    else:
        print("The folder for renders exists:", folder_path)   
    return folder_path


def setup_highway_env():
    return setup_env("highway-fast-v0")


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
                            "scaling": 25,
                            "lanes_count": 4
                         },
                     )
    env.reset()

    return env

class Results():
    def __init__(self):
        # Initialise all the tracking variables
        self.renders = np.zeros((1,750,2500,3))
        self.rewards, self.dones, self.truncateds, self.times = (np.array([[0]]) for _ in range(4))
        self.criticality = []
        self.peak_renderings = []

        # Dictionary of all variables to be tracked
        self.results_dict = {
            "renders": self.renders,
            "rewards": self.rewards, 
            "dones": self.dones, 
            "truncateds": self.truncateds,
            "times": self.times,
            "criticality": self.criticality,
            "peak_renderings": self.peak_renderings}
        
        self.path = None
        
    def __str__(self):
        self.return_average()
        return f"Number of episodes: {len(self.dones)-1}\nAvg rewards: {self.avg_rewards}\nDones: {list(self.dones[2:]).count([1])}/{len(self.dones)-1}\nTruncateds: {list(self.truncateds[2:]).count([1])}/{len(self.truncateds)-1}\nAvg Time: {self.avg_times}\nCriticalities: {self.criticality}"

    def return_average(self, merge=False):
        if merge:
            self.avg_rewards = {"general_reward": statistics.mean([reward[0]["general_reward"] for reward in self.rewards[2:]]),
                        "collision_reward": statistics.mean([reward[0]["collision_reward"] for reward in self.rewards[2:]]),
                        "right_lane_reward": statistics.mean([reward[0]["right_lane_reward"] for reward in self.rewards[2:]]),
                        "high_speed_reward": statistics.mean([reward[0]["high_speed_reward"] for reward in self.rewards[2:]]),
                        "lane_change_reward": statistics.mean([reward[0]["lane_change_reward"] for reward in self.rewards[2:]]),
                        "merging_speed_reward": statistics.mean([reward[0]["merging_speed_reward"] for reward in self.rewards[2:]])}
        else:
            self.avg_rewards = {"general_reward": statistics.mean([reward[0]["general_reward"] for reward in self.rewards[2:]]),
                        "collision_reward": statistics.mean([reward[0]["collision_reward"] for reward in self.rewards[2:]]),
                        "right_lane_reward": statistics.mean([reward[0]["right_lane_reward"] for reward in self.rewards[2:]]),
                        "high_speed_reward": statistics.mean([reward[0]["high_speed_reward"] for reward in self.rewards[2:]]),
                        "on_road_reward": statistics.mean([reward[0]["on_road_reward"] for reward in self.rewards[2:]])}        
        
        self.avg_times = statistics.mean([i[0] for i in self.times[2:]])
        

    def append(self, data):
        """
        Appends the data to be tracked.
        :param data: should be in order [renders, rewards, dones, truncateds, times, criticality, peak_renderings]
        """
        # self.renders = np.append(self.renders, np.array(data[0]).reshape(1,750,2500,3), axis=0)
        self.rewards = np.append(self.rewards,np.array(data[1]).reshape((1,1)), axis=0)
        self.dones = np.append(self.dones,np.array(data[2]).reshape((1,1)), axis=0)
        self.truncateds = np.append(self.truncateds,np.array(data[3]).reshape((1,1)), axis=0)
        self.times = np.append(self.times,np.array(data[4]).reshape((1,1)), axis=0) 
        self.criticality.append(data[5])
        self.peak_renderings.append(data[6])
        
    def save(self, merge=False, copy_num=None):
        # Reinitialising dictionary of all variables to be tracked
        self.results_dict = {
            "renders": self.renders,
            "rewards": self.rewards, 
            "dones": self.dones, 
            "truncateds": self.truncateds,
            "times": self.times,
            "criticality": self.criticality}
                
        # Step 1: Determine the maximum length
        max_length = max(len(episode) for episode in self.criticality)

        # Step 2: Pad each array to the maximum length
        padded_episodes = [
            np.pad(episode, (0, max_length - len(episode)), mode='constant', constant_values=np.nan)
            for episode in self.criticality
        ]

        # Step 3: Convert to a 2D NumPy array
        self.results_dict["criticality"] = np.array(padded_episodes)

        # Saving the np array of all the last states(first array is zeroes)
        if merge:
            for var in self.results_dict.keys():
                np.save(os.path.join(str(results_path(find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn")))[:-4]+"_merge.zip", f"{var}"), self.results_dict[var])
        else:    
            for var in self.results_dict.keys():
                np.save(os.path.join(results_path(find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn")), f"{var}"), self.results_dict[var])

        # Prepare to save peak renderings in a structured directory
        model_name = os.path.basename(find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn"))
        render_dir = os.path.join("criticality_renderings", model_name)
        
        if not os.path.exists(render_dir):
            os.makedirs(render_dir)

        # Save each episode's peak renderings in its own folder
        for episode_idx, episode_renderings in enumerate(self.peak_renderings):
            episode_dir = os.path.join(render_dir, f"episode_{episode_idx}")
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)

            # Save each rendering in the episode's directory, using the timestamp as filename
            for timestamp, rendering in tqdm.tqdm(episode_renderings):
                filename = f"{timestamp:.2f}.png"  # Save as PNG
                file_path = os.path.join(episode_dir, filename)
                
                # Convert numpy array to image and save as PNG
                image = Image.fromarray((rendering * 255).astype(np.uint8))  # Assuming rendering is in [0,1] range
                image.save(file_path)
        
    def load(self, model_path=find_model_path(iter=iterations, last=True, copy_num=None, model_type="dqn"), merge=False):
        self.path = results_path(model_path)
        files = os.listdir(self.path)
        print(files)
        self.dones = np.load(os.path.join(self.path, 'dones.npy'), allow_pickle=True)
        self.renders = np.load(os.path.join(self.path, 'renders.npy'), allow_pickle=True)
        self.rewards = np.load(os.path.join(self.path, 'rewards.npy'), allow_pickle=True)
        self.times = np.load(os.path.join(self.path, 'times.npy'), allow_pickle=True)
        self.truncateds = np.load(os.path.join(self.path, 'truncateds.npy'), allow_pickle=True)
        self.criticality = np.load(os.path.join(self.path, 'criticality.npy'), allow_pickle=True)

        self.return_average(merge)
              
    



