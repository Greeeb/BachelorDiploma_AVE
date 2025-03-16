from Functions import *

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

iterations = 100000
merge = False
compare = False
copy_num=20

def main():
    model_names = list()

    # Initialise results class for the first Highway model
    results_1 = Results()
    model_path = find_model_path(iter=1000, last=True, copy_num=copy_num, model_type="dqn")
    results_1.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))

    # Initialise results class for the second Merge model
    results_2 = Results()
    model_path = find_model_path(iter=10000, last=True, copy_num=copy_num, model_type="dqn")
    results_2.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))
    
    # Initialise results class for the second Merge model
    results_3 = Results()
    model_path = find_model_path(iter=50000, last=True, copy_num=copy_num, model_type="dqn")
    results_3.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))
    
    # Initialise results class for the second Merge model
    results_4 = Results()
    model_path = find_model_path(iter=100000, last=True, copy_num=copy_num, model_type="dqn")
    results_4.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))
    
    # Initialise results class for the second Merge model
    results_5 = Results()
    model_path = find_model_path(iter=150000, last=True, copy_num=copy_num, model_type="dqn")
    results_5.load(model_path, merge=merge)
    model_names.append(model_path.replace(str(os.path.join(os.path.abspath(os.curdir), "BachelorDiploma_AVE", "models")), ""))

    # Extract and calculate metrics for each model
    results = [results_1, results_2, results_3, results_4, results_5]
    iterations = [1000, 10000, 50000, 100000, 150000]
    
    # Dictionary to store training times and iterations per second
    training_stats = {
        "iterations": iterations,  # Model iterations (1000, 10000, etc.)
        "time_taken": [],          # Time taken to train (to be filled in manually or programmatically)
    }

    # Example of filling values (replace with actual training data)
    # These values are placeholders; replace with actual measured data
    training_stats["time_taken"] = [116, 1207, 5857, 11083, 17178]  # Time in seconds

    # Calculate relative time taken for each model
    relative_times = np.array(training_stats["time_taken"])

    # Initialize arrays for metrics
    avg_times = []
    avg_rewards = []
    collision_rates = []

    # Extract and calculate metrics for each model
    for result in results:
        result.return_average()
        avg_times.append(result.avg_times)
        avg_rewards.append(result.avg_rewards["general_reward"])
        collision_count = sum(result.dones)[0]  # Count collisions
        total_episodes = len(result.dones) - 1
        collision_rate = collision_count / total_episodes
        collision_rates.append(collision_rate)

    # Create the plot with a clean style
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Training Time
    ax1.plot(iterations, training_stats["time_taken"], marker='o', label='Training Time (s)', color='b', linewidth=2)
    ax1.set_xlabel('Number of Iterations', fontsize=12)
    ax1.set_ylabel('Training Time (seconds)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(visible=True, which='major', linestyle='--', linewidth=0.5)

    # Plot General Reward on the twin y-axis
    ax2 = ax1.twinx()
    ax2.plot(iterations, avg_rewards, marker='s', label='General Reward', color='g', linewidth=2)
    ax2.set_ylabel('General Reward', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Plot Relative Training Time on the secondary x-axis
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    
    # Update and style the table
    collision_table_data = {
        'Iterations': iterations,
        'Average Time (s)': avg_times,
        'Collision Rate': [f"{rate:.2%}" for rate in collision_rates],  # Display as percentages
        'Training Time (s)': training_stats["time_taken"],
    }
    collision_table_values = list(zip(
        collision_table_data['Iterations'],
        collision_table_data['Average Time (s)'],
        collision_table_data['Collision Rate'],
        collision_table_data['Training Time (s)']
    ))
    # collision_table = plt.table(
    #     cellText=collision_table_values,
    #     colLabels=list(collision_table_data.keys()),
    #     cellLoc='center',
    #     loc='bottom',
    #     bbox=[0.1, -0.5, 0.8, 0.3],
    #     fontsize = 12,
    # )

    # # Table Style Adjustments
    # collision_table.auto_set_font_size(False)
    # collision_table.set_fontsize(10)
    # collision_table.auto_set_column_width(col=list(range(len(collision_table_data.keys()))))

    # # Add all legends to the upper-left corner
    # lines_labels = [
    #     ax1.get_legend_handles_labels(),
    #     ax2.get_legend_handles_labels(),
    #     ax3.get_legend_handles_labels()
    # ]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, loc='center right', fontsize=10, frameon=True, shadow=True, title='Legend', title_fontsize=12, bbox_to_anchor=(0.91,0.51))

    # Final layout adjustments
    plt.title('Model Performance Metrics and Training Times Across Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    
if __name__=="__main__":
    main()