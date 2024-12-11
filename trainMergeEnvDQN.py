from stable_baselines3 import DQN

from Functions import *


iterations = 100000
seed = 100

def main(iter_load = iterations, iterations=iterations, copy_num=5, seed=seed):
    env = setup_merge_env()
    
    # Loading the model
    model_path = find_model_path(iter=iter_load, last=True, copy_num=copy_num, model_type="dqn")
    
    model = DQN('MlpPolicy', env=env, exploration_fraction=0.1, seed=seed, # make sure to keep seed same
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=500,
                    device="cuda:1"
        ).load(model_path, env=env)
    
    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=iterations, last=True, copy_num=copy_num, model_type="dqn"))

if __name__=="__main__":
    main()