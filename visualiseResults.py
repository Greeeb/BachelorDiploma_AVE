from Functions import *

from matplotlib import pyplot as plt

import numpy as np

iterations = 50000


def main():
    model_names = list()

    # Initialise results class for the first Highway model
    results_highway = Results()

    # Load results from file in saveResults
    model_path = find_model_path(iter=50000, last=True, copy_num=0, model_type="dqn") # TODO check the copy number
    results_highway.load(model_path)
    print(results_highway)

    model_names.append(model_path.replace(str(os.path.join(os.curdir, "models")), ""))


    # Initialise results class for the second Merge model
    results_merge = Results()

    # Load results from file in saveResults
    model_path = find_model_path(iter=50000, last=True, copy_num=2, model_type="dqn") # TODO check the copy number
    results_merge.load(model_path)
    print(results_merge)

    model_names.append(model_path.replace(str(os.path.join(os.curdir, "models")), ""))


    avg_rewards = { "general_reward": [],
                    "collision_reward": [],
                    "right_lane_reward": [],
                    "high_speed_reward": [],
                    "on_road_reward":[]}
    for key in avg_rewards.keys():
        # Appending rewards to the dictionary with lists for each category
        avg_rewards[key].append(results_highway.avg_rewards[key])
        avg_rewards[key].append(results_merge.avg_rewards[key])

    # Combine dones into one list
    dones = [results_highway.dones.sum(), results_merge.dones.sum()]

    # Combine truncateds into one list
    truncateds = [results_highway.truncateds.sum(), results_merge.truncateds.sum()]

    # Combine all the average times into one list
    avg_times = [results_highway.avg_times, results_merge.avg_times]


    # Plot the data

    # 1. Bar chart for average rewards per model with values on top
    plt.figure(figsize=(20, 12))
    x = np.arange(len(model_names))
    width = 0.15  # width of the bars

    for i, (key, values) in enumerate(avg_rewards.items()):
        # Plot each set of bars with an offset
        bars = plt.bar(x + i * width, values, width=width, label=key)
        
        # Add the values on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, yval + 0.5,  # Positioning the text
                f'{yval:.2f}',  # Formatting the text to 2 decimal places
                ha='center', va='bottom', fontsize=9
            )

    plt.xlabel('Model')
    plt.ylabel('Average Rewards')
    plt.title('Average Rewards by Model')
    plt.xticks(x + width * 2, model_names)
    plt.legend()
    plt.tight_layout()
    #
    # Set up a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # Bar chart for General Reward
    axes[0].bar(model_names, avg_rewards['general_reward'], color='skyblue')
    axes[0].set_title('General Reward by Model')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('General Reward')

    # Bar chart for Dones
    axes[1].bar(model_names, dones, color='salmon')
    axes[1].set_title('Dones by Model')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Dones')

    # Bar chart for Avg Time
    axes[2].bar(model_names, avg_times, color='lightgreen')
    axes[2].set_title('Average Time by Model')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('Avg Time (seconds)')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Stacked bar chart for Dones and Truncateds
    plt.figure(figsize=(20, 12))
    plt.bar(model_names, dones, label='Dones')
    plt.bar(model_names, truncateds, bottom=dones, label='Truncateds')
    plt.xlabel('Model')
    plt.ylabel('Episode Count')
    plt.title('Dones and Truncated Episodes by Model')
    plt.legend()
    plt.tight_layout()
    plt.show()

    




if __name__=="__main__":
    main()