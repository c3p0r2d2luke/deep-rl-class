import os
from pyvirtualdisplay import Display
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub

# 1. Start Virtual Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. Setup Environments
env_id = "LunarLander-v2"
env = make_vec_env(env_id, n_envs=16)

# 3. Model Definition
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)

# 4. Train (500,000 steps will get it consistently above +200 reward)
model.learn(total_timesteps=500000)

model_name = "ppo-LunarLander-v2"
model.save(model_name)

# 5. Create Evaluation Environment (MISSING LINE)
eval_env = gym.make(env_id)

# 6. Evaluate
mean_reward, std_reward = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes=10, 
    deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 7. Push to Hub
repo_id = "Frenchwizz/ppo-LunarLander-v2"
package_to_hub(
    model=model,
    model_name=model_name,
    model_architecture="PPO",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=repo_id,
    commit_message="Upload PPO LunarLander-v2 agent"
)

