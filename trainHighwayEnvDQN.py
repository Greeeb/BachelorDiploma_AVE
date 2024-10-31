from stable_baselines3 import DQN

from Functions import *


iterations = 50000

env = setup_highway_env()

model = DQN('MlpPolicy', env=env, exploration_fraction=0.7, seed=100, # make sure to keep seed same
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1, 
                tensorboard_log='logs/highway_dqn/')

model.learn(iterations, progress_bar=True)
model.save(find_model_path(iter=iterations, last=True, model_type="dqn"))
