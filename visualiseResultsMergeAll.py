from Functions import *

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

iterations = 100000
merge = True
copy_num=0

def main(copy_num=copy_num):
    model_names = list()
    
    # Initialise results class for the first Merge model
    results_1 = Results()
    model_path = find_model_path(iter=100000, last=True, copy_num=copy_num, model_type="dqn")
    results_1.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))

    # Initialise results class for the second Merge model
    results_2 = Results()
    model_path = find_model_path(iter=10000, last=True, copy_num=copy_num, model_type="dqn")
    results_2.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))
    
    # Initialise results class for the second Merge model
    results_3 = Results()
    model_path = find_model_path(iter=1000, last=True, copy_num=copy_num, model_type="dqn")
    results_3.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))

    # Define the reward types
    reward_types = ["general_reward", "collision_reward", "right_lane_reward", "high_speed_reward", "lane_change_reward", "merging_speed_reward"]

    # Bar chart for average rewards per model
    avg_rewards = {reward: [] for reward in reward_types}
    for reward in reward_types:
        avg_rewards[reward].append(results_1.avg_rewards[reward])
        avg_rewards[reward].append(results_2.avg_rewards[reward])
        avg_rewards[reward].append(results_3.avg_rewards[reward])


    plt.figure(figsize=(14, 8))
    x = np.arange(len(model_names))
    width = 0.15

    for i, (key, values) in enumerate(avg_rewards.items()):
        bars = plt.bar(x + i * width, values, width=width, label=key)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('Model')
    plt.ylabel('Average Rewards')
    plt.title('Average Rewards by Model')
    plt.xticks(x + width * 2, model_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Rewards.png")

    # Dones and truncateds bar chart
    dones = [results_1.dones.sum(), results_2.dones.sum()]
    truncateds = [results_1.truncateds.sum(), results_2.truncateds.sum()]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, dones, label='Dones')
    plt.bar(model_names, truncateds, bottom=dones, label='Truncateds')
    plt.xlabel('Model')
    plt.ylabel('Episode Count')
    plt.title('Dones and Truncated Episodes by Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Dones-Truncateds.png')

    # Criticality plots for first 10 episodes
    model1_data = results_1.criticality
    model2_data = results_2.criticality
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for i, ax in enumerate(axes.flat):
        ax.plot(np.ma.masked_invalid(model1_data[i]), label='Model 1', color='blue')
        ax.plot(np.ma.masked_invalid(model2_data[i]), label='Model 2', color='red')
        ax.set_title(f'Episode {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.savefig("Criticality_Comparison_First10.png")

    # Criticality distribution plot with logarithmic scale on the y-axis
    plt.figure(figsize=(12, 6))
    criticality_highway = np.nan_to_num(model1_data).flatten()
    criticality_merge = np.nan_to_num(model2_data).flatten()
    sns.histplot(criticality_highway, label="Model 1", color="blue", kde=True, log_scale=(False, True))
    sns.histplot(criticality_merge, label="Model 2", color="red", kde=True, log_scale=(False, True))
    plt.title("Criticality Distribution by Frequency (Logarithmic Scale)")
    plt.xlabel("Criticality")
    plt.ylabel("Frequency (Log Scale)")
    plt.legend()
    plt.savefig("Criticality_Distribution_Log.png")

    # Reward type comparison across episodes
    fig, axes = plt.subplots(len(reward_types), 1, figsize=(10, 20))
    for i, reward_type in enumerate(reward_types):
        axes[i].plot([reward[0][reward_type] for reward in results_1.rewards[1:101]], label="Model 1", color="blue")
        axes[i].plot([reward[0][reward_type] for reward in results_2.rewards[1:101]], label="Model 2", color="red")
        axes[i].set_title(f"Reward: {reward_type}")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig("Reward_Comparison.png")

    # # Episode time comparison
    # episode_times_highway = results_1.times
    # episode_times_merge = results_2.times
    # avg_time_highway = np.mean(episode_times_highway)
    # avg_time_merge = np.mean(episode_times_merge)

    # plt.figure(figsize=(10, 6))
    # plt.plot(episode_times_highway[1:101], label="Model 1", color="blue")
    # plt.plot(episode_times_merge[1:101], label="Model 2", color="red")
    # plt.axhline(avg_time_highway, color="blue", linestyle="--", label=f"Model 1 Avg: {avg_time_highway:.2f}s")
    # plt.axhline(avg_time_merge, color="red", linestyle="--", label=f"Model 2 Avg: {avg_time_merge:.2f}s")
    # plt.xlabel("Episode")
    # plt.ylabel("Time (seconds)")
    # plt.title("Episode Time Comparison by Model")
    # plt.legend()
    
    models = [""]
    # Plot average and variance for each model
    for i, model in enumerate(models):
        plt.plot(x, average_times[i], label=f"{model} Average", linewidth=2)
        plt.plot(x, variance_times[i], linestyle="--", label=f"{model} Variance", linewidth=1)

    # Add labels, legend, and title
    plt.xlabel("Time Step")
    plt.ylabel("Time (ms)")
    plt.title("Average and Variance of Times for Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Episode_Times.png")

if __name__=="__main__":
    main()
