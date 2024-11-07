from stable_baselines3 import DQN

from Functions import *


iterations = 100000

def main():
    env = setup_merge_env()
    
    rendering = env.render()
    image = Image.fromarray((rendering * 255).astype(np.uint8))  # Assuming rendering is in [0,1] range
    image.save("file_path.png")
    
    model = DQN('MlpPolicy', env=env, exploration_fraction=0.7, seed=100, # make sure to keep seed same
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=500,
                    device="cuda:1"
        )
    
    model.learn(iterations, progress_bar=True)
    model.save(find_model_path(iter=iterations, last=True, copy_num=3, model_type="dqn"))

if __name__=="__main__":
    main()