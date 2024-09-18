import gymnasium as gym
import highway_env
from highway_env.envs.common.observation import OccupancyGridObservation 
from highway_env.envs.common.action import ContinuousAction
from stable_baselines3 import SAC, DQN

gym.register_envs(highway_env)

env = gym.make('highway-fast-v0', 
               render_mode='rgb_array',
               config={
                    "screen_width": 2500, 
                    "screen_height": 750,
                    "scaling": 25
                 }
             )

model = DQN('MlpPolicy', env=env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=100,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.3,
                verbose=1,
                tensorboard_log='highway_dqn/')
model.learn(int(4e3), progress_bar=True)

model.save("model_dqn")


for episode in range(100):
    (obs, info), done, truncated = env.reset(), False, False
    while not (done or truncated):
        print(obs)
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, truncated, info = env.step(int(action))
        
        env.render()