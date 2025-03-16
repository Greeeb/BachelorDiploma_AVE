from Functions import *

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

iterations = 100000
merge = True
copy_num=1

def main(copy_num=copy_num):
    model_names = list()
    results = list()
    for copy_num in range(10):
        # Initialise results class for the first Merge model
        results_1 = Results()
        model_path = find_model_path(iter=100000, last=True, copy_num=copy_num, model_type="dqn")
        results_1.load(model_path, merge=merge)
        

        # Initialise results class for the second Merge model
        results_2 = Results()
        model_path = find_model_path(iter=10000, last=True, copy_num=copy_num, model_type="dqn")
        results_2.load(model_path, merge=merge)
        
        
        # Initialise results class for the second Merge model
        results_3 = Results()
        model_path = find_model_path(iter=1000, last=True, copy_num=copy_num, model_type="dqn")
        results_3.load(model_path, merge=merge)
        
        
        results.append([results_1, results_2, results_3])

    model_names.append("Pure Merge Model")
    model_names.append("Highway-Merge Model")
    model_names.append("Highway-Merge Criticals")
    
    # Define the reward types
    reward_types = ["general_reward", "collision_reward", "right_lane_reward", "high_speed_reward", "lane_change_reward", "merging_speed_reward"]
    colors = {"Pure Merge Model": "b", "Highway-Merge Model": "orange", "Highway-Merge Criticals": "g"}


    # Calculate average times for each model type across experiments
    avg_times = {
        "Pure Merge Model": [],
        "Highway-Merge Model": [],
        "Highway-Merge Criticals": []
    }

    for experiment in results:
        avg_times["Pure Merge Model"].append(experiment[0].avg_times)
        avg_times["Highway-Merge Model"].append(experiment[1].avg_times)
        avg_times["Highway-Merge Criticals"].append(experiment[2].avg_times)

    plt.figure(figsize=(10, 6))
    x = np.arange(1, 11)
    width = 0.3  # Bar width

    print(avg_times)
    
    for i, model in enumerate(model_names):
        plt.bar(x + i * width, avg_times[model], width, label=model, color=colors[model], alpha=0.7)
        avg_time = np.mean(avg_times[model])
        plt.axhline(avg_time, color=colors[model], linestyle="dashed", label=f"{model} (Avg)")
        if model=='Highway-Merge Criticals':
            plt.text(10.8, avg_time-0.03, f"{avg_time:.2f}", fontsize=10)
        else:
            plt.text(10.8, avg_time+0.005, f"{avg_time:.2f}", fontsize=10)

    plt.xlabel("Experiment")
    plt.ylabel("Average Episode Duration")
    plt.title("Average Episode Duration per Experiment")
    plt.xticks(x + width, range(1, 11))
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # Calculate average rewards and standard deviations for each model type across experiments
    avg_rewards_all = {
        "Pure Merge Model": {reward: [] for reward in reward_types},
        "Highway-Merge Model": {reward: [] for reward in reward_types},
        "Highway-Merge Criticals": {reward: [] for reward in reward_types}
    }

    std_rewards_all = {
        "Pure Merge Model": {reward: [] for reward in reward_types},
        "Highway-Merge Model": {reward: [] for reward in reward_types},
        "Highway-Merge Criticals": {reward: [] for reward in reward_types}
    }

    # Aggregate rewards and compute means and standard deviations
    for experiment in results:
        for i, model in enumerate(avg_rewards_all.keys()):
            for reward in reward_types:
                avg_rewards_all[model][reward].append(experiment[i].avg_rewards[reward])

    for model in avg_rewards_all:
        for reward in reward_types:
            std_rewards_all[model][reward] = np.std(avg_rewards_all[model][reward])
            avg_rewards_all[model][reward] = np.mean(avg_rewards_all[model][reward])
            

    # Convert general rewards into a bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.2  # Width of each bar
    x = np.arange(len(results))  # X-axis positions for experiments

    for i, (model, color) in enumerate(avg_rewards_all.items()):
        general_rewards = [experiment[i].avg_rewards["general_reward"] for experiment in results]
        
        # Plot bar chart
        plt.bar(x + i * bar_width, general_rewards, width=bar_width, label=model, color=colors[model], alpha=0.7)
        
        # Calculate and plot average line
        avg_general_reward = np.mean(general_rewards)
        plt.axhline(y=avg_general_reward, color=colors[model], linestyle='--')
        plt.text(len(results), avg_general_reward+0.1, f'{avg_general_reward:.2f}')

    # Formatting
    plt.xlabel('Experiment')
    plt.ylabel('General Reward')
    plt.title('General Reward per Experiment for Each Model')
    plt.xticks(x + bar_width, range(1, len(results) + 1))  # Align x-ticks
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    

    x = np.arange(len(reward_types))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (model, color) in enumerate(colors.items()):
        means = [avg_rewards_all[model][reward] for reward in reward_types]
        stds = [std_rewards_all[model][reward] for reward in reward_types]
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=6, color=color, label=model, alpha=0.7)

        # Add data labels above the error bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.2, f'{mean:.2f}',
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Reward Types', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    ax.set_title('Average Rewards by Model and Reward Type (Averaged Over Experiments)', fontsize=16)

    ax.set_xticks(x + width)
    ax.set_xticklabels(reward_types, rotation=45, ha="right", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
   
    
    # Collect criticality values for each model type across experiments
    criticality_values = {
        "Pure Merge Model": [],
        "Highway-Merge Model": [],
        "Highway-Merge Criticals": []
    }

    # for experiment in results:
    #     criticality_values["Pure Merge Model"].extend(experiment[0].criticality.flatten())
    #     criticality_values["Highway-Merge Model"].extend(experiment[1].criticality.flatten())
    #     criticality_values["Highway-Merge Criticals"].extend(experiment[2].criticality.flatten())
    
    # # Plot the smooth criticality appearance count graph with logarithmic y-axis
    # plt.figure(figsize=(10, 6))

    # for model, values in criticality_values.items():
    #     # Remove NaN values (if any)
    #     values = np.array(values)
    #     values = values[~np.isnan(values)]  # Filter out NaN values
        
    #     # Use KDE for smooth curve
    #     sns.kdeplot(values, label=model, log_scale=(False, True))  # Log scale only on y-axis

    # plt.xlabel('Criticality Value')
    # plt.ylabel('Density (Log Scale)')
    # plt.title('Testing Graph of Smooth Distribution of Criticality Values (Logarithmic Y-axis)')
    # plt.legend()
    # plt.grid(True)
    
    # Calculate collision rates for each model type across experiments
    collision_rates = {
        "Pure Merge Model": [],
        "Highway-Merge Model": [],
        "Highway-Merge Criticals": []
    }

    for experiment in results:
        for i, model in enumerate(collision_rates.keys()):
            # Count the number of episodes with collisions
            collision_count = np.sum([1 for done in experiment[i].dones[2:] if done[0] == 1])
            total_episodes = len(experiment[i].dones[2:])
            collision_rates[model].append(collision_count / total_episodes)
    
    x = np.arange(1, 11)
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        plt.bar(x + i * width, collision_rates[model], width, label=model, color=colors[model], alpha=0.7)
        avg_rate = np.mean(collision_rates[model])
        plt.axhline(avg_rate, color=colors[model], linestyle="dashed", label=f"{model} (Avg)")
        plt.text(10.55, avg_rate + 0.01, f"{avg_rate:.2f}", fontsize=10, color="black")

    plt.xlabel("Experiment")
    plt.ylabel("Collision Rate")
    plt.title("Collision Rate per Experiment")
    plt.xticks(x + width, range(1, 11))
    plt.legend(loc="center right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # File paths and data collection
    base_path = r"C:\Users\memen\THI\BachelorDiploma_AVE\saveResults\Highway100000_accessed"
    critical_obs_counts = []
    traditional_steps = [10000] * 10  # Traditional TL step count

    for i in range(10):
        file_path = os.path.join(base_path, f"model_dqn_100000({i})", "crit_obs.npy")
        critical_obs_counts.append(len(np.load(file_path)) if os.path.exists(file_path) else 0)

    # Define x-axis positions
    x = np.arange(1, 11)  # Experiment numbers
    bar_width = 0.4  # Bar width

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, critical_obs_counts, width=bar_width, color=colors["Highway-Merge Criticals"], alpha=0.7, label="Critical TL Steps")
    plt.bar(x + bar_width / 2, traditional_steps, width=bar_width, color=colors["Highway-Merge Model"], alpha=0.7, label="Traditional TL Steps")

    # Compute and plot average line for critical observations
    avg_critical_obs = np.mean(critical_obs_counts)
    plt.axhline(avg_critical_obs, color="black", linestyle="dashed", label="Avg Critical Observations")
    plt.text(10.9, avg_critical_obs + 100, f"{avg_critical_obs:.2f}", fontsize=10, color="black")
    
    # Add text labels above bars
    for i, obs in enumerate(critical_obs_counts):
        plt.text(x[i] - bar_width / 2, obs + 100, f"{obs}", fontsize=10, color="black", ha='center', va='bottom')
    for i, step in enumerate(traditional_steps):
        plt.text(x[i] + bar_width / 2, step + 100, f"{step}", fontsize=10, color="black", ha='center', va='bottom')

    # Formatting
    plt.xlabel("Experiment")
    plt.ylabel("Number of Training Steps")
    plt.title("Comparison of Critical TL vs Traditional TL Training Step Number")
    plt.xticks(x)
    plt.legend(loc="center right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    
    plt.show()


if __name__=="__main__":
    main()
