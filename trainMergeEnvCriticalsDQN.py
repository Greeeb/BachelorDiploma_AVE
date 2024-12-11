from stable_baselines3 import DQN
import torch
from trainHighwayEnvDQN import main as trainHighway

from Functions import *


iter = 10000
criticals_file = "model_dqn_100000_crit"  # Folder name from the criticality_renderings folder
torch.cuda.set_device(1)
copy_num = 7
seed = 100


def main(copy_num=copy_num, iter_load=iter, criticals_file = criticals_file, seed=seed, save_copy=copy_num, iter_save=iter):
    env = setup_merge_env()

    results = Results()
    results.load(
        os.path.join(os.path.abspath(os.path.curdir), "saveResults", criticals_file)
        )
    critical_obs = results.crit_obs
    iterations = critical_obs.shape[0]
    print(iterations)
    
    
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
                    replay_buffer_class=CustomReplayBuffer,
                    replay_buffer_kwargs={
                        # 'observation_space': observation_space,
                        # 'buffer_size': 1000000,
                        # 'action_space': action_space,
                    }
        ).load(model_path, env=env)
    
    # critical_obs = np.concatenate([critical_obs, critical_obs], axis=0 )
    for obs in tqdm.tqdm(critical_obs):
        action, _ = model.predict(obs, deterministic=True)
        obs_next, reward, done, truncated, info = env.step(int(action))
        model.replay_buffer.add(obs=obs, next_obs=obs_next, reward=reward, done=done, action=action, infos=info)

    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=iter_save, last=True, copy_num=save_copy, model_type="dqn"))
    

if __name__=="__main__":
    main()