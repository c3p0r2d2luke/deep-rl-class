import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from fnaf_env import WebFNAFEnv

class Nights1And2Callback(BaseCallback):
    def __init__(self, save_path="ppo-fnaf-nights1-2", target_night2_wins=3, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.target_night2_wins = target_night2_wins
        self.night1_wins = 0
        self.night2_wins = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}])

        if len(dones) > 0 and dones and rewards >= 90.0:
            night = infos.get("night", 1)
            if night == 1:
                self.night1_wins += 1
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(
                        f"\n[Curriculum] Night 1 WIN #{self.night1_wins}! Advancing straight into Night 2... "
                        f"(Saved to {self.save_path}.zip at step {self.num_timesteps})\n",
                        flush=True,
                    )
            elif night == 2:
                self.night2_wins += 1
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(
                        f"\n[Curriculum] Night 2 WIN #{self.night2_wins}/{self.target_night2_wins}! "
                        f"Saved to {self.save_path}.zip (step {self.num_timesteps})\n",
                        flush=True,
                    )

                if self.night2_wins >= self.target_night2_wins:
                    if self.verbose > 0:
                        print(f"[Curriculum] Mastered Nights 1 & 2 ({self.target_night2_wins} Night 2 wins)! Stopping...", flush=True)
                    return False

        return True

base_env = WebFNAFEnv(url="http://localhost:8000/FNAF.html")
env = Monitor(base_env)

# Load your pre-trained Night 1 weights so it starts with existing knowledge!
# Check for the latest Nights 1 & 2 checkpoint first, then fall back to Night 1
if os.path.exists("ppo-fnaf-nights1-2.zip"):
    pretrained_path = "ppo-fnaf-nights1-2.zip"
elif os.path.exists("ppo-fnaf-night1.zip"):
    pretrained_path = "ppo-fnaf-night1.zip"
else:
    pretrained_path = None
if os.path.exists(pretrained_path):
    print(f"Loading pre-trained Night 1 model from {pretrained_path}...")
    model = PPO.load(
        pretrained_path,
        env=env,
        learning_rate=0.0002,
        n_steps=64,
        batch_size=64,
        n_epochs=4,
        ent_coef=0.02,
        verbose=1,
    )
else:
    print("No ppo-fnaf-night1.zip found; initializing fresh CnnPolicy model...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=0.00025,
        n_steps=64,
        batch_size=64,
        n_epochs=4,
        ent_coef=0.02,
        verbose=1,
    )

print("Starting curriculum training: Night 1 -> Night 2 (auto-saving on every win, stopping at 3 Night 2 wins)...")
callback = Nights1And2Callback(save_path="ppo-fnaf-nights1-2", target_night2_wins=3)

try:
    model.learn(
        total_timesteps=100000,
        log_interval=1,
        callback=callback,
        reset_num_timesteps=False,  # Preserves your existing step count from Night 1 training
    )
except KeyboardInterrupt:
    print("\nTraining stopped manually with Ctrl+C!")
finally:
    model.save("ppo-fnaf-nights1-2")
    env.close()
    print(
        f"Finished with {callback.night1_wins} Night 1 wins and {callback.night2_wins} Night 2 wins. "
        f"Saved to ppo-fnaf-nights1-2.zip."
    )
