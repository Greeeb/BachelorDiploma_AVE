import gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from stable_baselines3 import DQN
from Functions import *

# Define the environment
env = setup_merge_env()  # Ensure this environment is installed and registered

# Load the trained DQN model
model_path = "models/model_dqn_100000(2).zip"  # Change to your model's path
model = DQN.load(model_path)

# Store criticality values and frames
criticality_values = []
frames = []

# Function to compute criticality (Example: variance of Q-values)
def compute_criticality(q_values):
    return np.var(q_values)

# Run the model for 10 episodes
for episode in range(2):
    state = env.reset(seed=episode*10)[0]
    done = False

    while not done:
        # Get Q-values for all actions
        q_values = model.q_net(torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)).tolist()[0]
        
        # Compute criticality
        criticality = compute_criticality(q_values)
        criticality_values.append(criticality)

        # Render and store frame
        frames.append(env.render())

        # Select action using the trained policy
        action, _ = model.predict(state, deterministic=True)

        # Step the environment
        state, reward, done, _, _ = env.step(action)

env.close()

# Convert criticality values to numpy array
criticality_values = np.array(criticality_values)

# Find local peaks in criticality
local_peaks = argrelextrema(criticality_values, np.greater)[0]  # Get indices of local peaks

# Plot criticality values with peaks marked
plt.figure(figsize=(10, 5))
plt.plot(criticality_values, label="Criticality Value")
plt.scatter(local_peaks, criticality_values[local_peaks], color='red', label="Local Peaks")
plt.xlabel("Time Step")
plt.ylabel("Criticality")
plt.title("Criticality Values Over Time")
plt.legend()

# Show renders at critical peaks
for peak in local_peaks:
    plt.figure()
    plt.imshow(frames[peak])
    plt.axis("off")
    plt.title(f"Render at Criticality Peak {peak}")
    
plt.show()
