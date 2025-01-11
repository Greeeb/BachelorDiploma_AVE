from Functions import *

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

iterations = 100000
merge = True
copy_num=0

def main(copy_num=copy_num):
    model_names = ["Model Merge 100000", "Model Merge 10000", "Model Merge Critical"]
    results = [[],[],[]]
    
    for i in range(10):
        copy_num=i
        
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
        
        results[0].append(results_1)
        results[1].append(results_2)
        results[2].append(results_3)
    
    x = np.arange(10)
    # Plot average and variance for each model
    for i, model in enumerate(model_names):
        plt.plot(x, [np.mean(results[i][index].times) for index in range(10)], label=f"{model} Average", linewidth=2)
        # plt.plot(x, [np.sqrt(np.var(results[i][index].times)) for index in range(10)], linestyle="--", label=f"{model} Stand. Deviation", linewidth=1)

    # Add labels, legend, and title
    plt.xlabel("Experiment")
    plt.ylabel("Time (ms)")
    plt.title("Average and Standart Deviation of Times for Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r"Figures/Times_deviation.png")
    
    
    for i, model in enumerate(model_names):
        plt.plot(x, [np.mean(results[i][index].criticality) for index in range(10)], label=f"{model} Average Criticality", linewidth=2)
        plt.plot(x, [np.sqrt(np.var(results[i][index].criticality)) for index in range(10)], linestyle="--", label=f"{model} Stand. Deviation", linewidth=1)

    plt.xlabel("Experiment")
    plt.ylabel("Criticality")
    plt.title("Criticality Over Experiments for Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r"Figures/Criticality_deviation.png")
    
    
    success_rates = []
    failure_rates = []
    truncation_rates = []

    for i, model in enumerate(model_names):
        success_rates.append([np.sum(results[i][index].dones == 1) for index in range(10)])
        failure_rates.append([np.sum(results[i][index].dones == 0) for index in range(10)])
        truncation_rates.append([np.sum(results[i][index].truncateds == 1) for index in range(10)])

    x = np.arange(10)
    for i, model in enumerate(model_names):
        plt.plot(x, success_rates[i], label=f"{model} Success Rate", linewidth=2)
        plt.plot(x, failure_rates[i], label=f"{model} Failure Rate", linewidth=2, linestyle="--")
        plt.plot(x, truncation_rates[i], label=f"{model} Truncation Rate", linewidth=2, linestyle=":")

    plt.xlabel("Experiment")
    plt.ylabel("Rate")
    plt.title("Success, Failure, and Truncation Rates for Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r"Figures/Success_failure_rates.png")

    # for i, model in enumerate(model_names):
    #     criticalities = [np.mean(results[i][index].criticality) for index in range(10)]
    #     rewards = [np.mean(results[i][index].rewards[""]) for index in range(10)]
    #     plt.scatter(criticalities, rewards, label=f"{model}", alpha=0.7)

    # plt.xlabel("Average Criticality")
    # plt.ylabel("Average Reward")
    # plt.title("Criticality vs Reward for Models")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(r"Figures/Criticality_vs_Reward.png")


    time_data = []

    for i, model in enumerate(model_names):
        model_times = [results[i][index].times.flatten() for index in range(10)]
        time_data.append(np.concatenate(model_times))

    plt.boxplot(time_data, labels=model_names)
    plt.xlabel("Model")
    plt.ylabel("Execution Time (ms)")
    plt.title("Execution Time Distribution Across Models")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r"Figures/Time_distribution.png")
    
    
    import seaborn as sns

    for i, model in enumerate(model_names):
        crit_obs = np.vstack([results[i][index].crit_obs for index in range(10)])  # Combine all critical observations
        heatmap_data = np.mean(crit_obs, axis=0)  # Average over all episodes

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap="coolwarm", annot=False)
        plt.title(f"Critical Observations Heatmap - {model}")
        plt.tight_layout()
        plt.savefig(rf"Figures/Crit_Obs_Heatmap_{model.replace(' ', '_')}.png")


    # Define reward components to be visualized
    components = ['general_reward', 'collision_reward', 'right_lane_reward', 'high_speed_reward', 'lane_change_reward']

    # Initialize a dictionary to store average rewards for each model
    avg_rewards = {model: {comp: [] for comp in components} for model in model_names}

    # Collect the merged average rewards for each experiment
    for i, model in enumerate(model_names):
        for experiment_idx in range(10):
            # Ensure merge=True when calling return_average for each experiment individually
            results[i][experiment_idx].return_average(merge=True)  
            # Append the average reward for each component
            for comp in components:
                avg_rewards[model][comp].append(results[i][experiment_idx].avg_rewards[comp])

    # Prepare for plotting the stacked bar chart
    x = np.arange(10)  # Number of experiments
    width = 0.2  # Bar width

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each model's reward breakdown
    for i, model in enumerate(model_names):
        bottom = np.zeros(10)  # Start from zero for each component
        for comp in components:
            # Plot the mean of rewards for each component across the experiments
            plt.bar(- width + x + i * width, avg_rewards[model][comp], width, bottom=bottom, label=f"{model} - {comp}")
            bottom += avg_rewards[model][comp]  # Add current component to bottom for stacking

    # Add labels, title, legend, and grid
    plt.xlabel("Experiment")
    plt.ylabel("Average Reward")
    plt.title("Reward Breakdown by Component (Merged) for Models")
    # plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()

    # Save the figure
    plt.savefig(r"Figures/Reward_breakdown_merged.png")


if __name__=="__main__":
    main()
