import time
from stable_baselines3 import PPO
from minesweeper_env import FastWebMinesweeperEnv

# Create fast environment instance
env = FastWebMinesweeperEnv(board_size=(9, 9))

# Train with MlpPolicy on numerical grid state
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    ent_coef=0.05,
    verbose=1
)

print("Waiting 5 seconds before starting...")
time.sleep(5)

print("Starting high-speed training...")
model.learn(total_timesteps=50000)
model.save("ppo-web-minesweeper")
env.close()
