import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from minesweeper_env import FastWebMinesweeperEnv

# 1. Custom 2D CNN so the network sees 3x3 neighbor relationships directly
class MinesweeperCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 channels * 9 height * 9 width = 5184 flattened features
        n_flatten = 64 * 9 * 9

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # Normalize board values (-1 to 8) and add channel dim -> (batch, 1, 9, 9)
        x = (obs.float() + 1.0) / 9.0
        x = x.unsqueeze(1)
        return self.linear(self.cnn(x))

# 2. Action mask function: only allow clicking covered tiles (-1)
def mask_fn(env):
    obs, _, _ = env.unwrapped._get_game_state()
    mask = (obs.flatten() == -1)
    if not np.any(mask):
        mask = np.ones(81, dtype=bool)
    return mask

# 3. Wrap environment with ActionMasker
base_env = FastWebMinesweeperEnv(board_size=(9, 9))
env = ActionMasker(base_env, mask_fn)

# 4. Train MaskablePPO with the 2D CNN extractor
policy_kwargs = dict(
    features_extractor_class=MinesweeperCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    ent_coef=0.01,
    verbose=1,
)

print("Starting spatial CNN + Action-Masked training...")
model.learn(total_timesteps=50000)
model.save("maskable-cnn-minesweeper")
env.close()
