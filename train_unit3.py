from pyvirtualdisplay import Display
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from huggingface_sb3 import package_to_hub

# 1. Start Virtual Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. Reward Shaping Wrapper
class MountainCarRewardWrapper(gym.RewardWrapper):
    """
    Gives additional reward based on position and velocity 
    to encourage the agent to build momentum up the hill.
    """
    def reward(self, reward):
        position, velocity = self.env.unwrapped.state
        # Reward the car for being high up the right hill or having high speed
        shaped_reward = reward + 10 * abs(velocity) + (position + 0.5)
        return shaped_reward

# 3. Create Environment
env_id = "MountainCar-v0"
env = gym.make(env_id)
env = MountainCarRewardWrapper(env)

# 4. Instantiate DQN (Off-policy DQN explores sparse rewards better than PPO)
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=250,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1
)

# 5. Train Agent
model.learn(total_timesteps=120000)

model_name = "dqn-MountainCar-v0"
model.save(model_name)

# 6. Evaluate on Unwrapped Environment (Standard -200 to 0 scale)
eval_env = gym.make(env_id, render_mode="rgb_array")
mean_reward, std_reward = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes=10, 
    deterministic=True
)
print(f"Mean reward on original env: {mean_reward:.2f} +/- {std_reward:.2f}")

# 7. Push Solved Model to Hub
package_to_hub(
    model=model,
    model_name=model_name,
    model_architecture="DQN",
    env_id=env_id,
    eval_env=eval_env,
    repo_id="Frenchwizz/dqn-MountainCar-v0",
    commit_message="Upload DQN MountainCar-v0 agent with reward shaping"
)