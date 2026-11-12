from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from fnaf_env import WebFNAFEnv

class WinSaveAndStopCallback(BaseCallback):
    def __init__(self, save_path="ppo-fnaf-night1", target_wins=3, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.target_wins = target_wins
        self.wins = 0

    def _on_step(self) -> bool:
        # Check if the latest step ended with the +100.0 6 AM win reward
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if len(dones) > 0 and dones[0] and rewards[0] >= 90.0:
            self.wins += 1
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(
                    f"\n[WinTracker] Night 1 WIN #{self.wins}/{self.target_wins}! "
                    f"Saved model to {self.save_path}.zip (step {self.num_timesteps})\n",
                    flush=True,
                )

            # Stop training once target_wins (3) is reached
            if self.wins >= self.target_wins:
                if self.verbose > 0:
                    print(f"[WinTracker] Reached {self.target_wins} wins! Stopping training...", flush=True)
                return False

        return True

base_env = WebFNAFEnv(url="http://localhost:8000/FNAF.html")
env = Monitor(base_env)

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

print("Starting Night 1 training (saving on each win, stopping at 3 wins)...")
callback = WinSaveAndStopCallback(save_path="ppo-fnaf-night1", target_wins=3)

try:
    model.learn(total_timesteps=50000, log_interval=1, callback=callback)
except KeyboardInterrupt:
    print("\nTraining stopped manually with Ctrl+C!")
finally:
    model.save("ppo-fnaf-night1")
    env.close()
    print(f"Finished with {callback.wins} wins. Model saved to ppo-fnaf-night1.zip.")
