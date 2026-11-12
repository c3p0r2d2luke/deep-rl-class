import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from playwright.sync_api import sync_playwright

class WebFNAFEnv(gym.Env):
    def __init__(self, url="http://localhost:8000/FNAF.html", frame_size=(84, 84)):
        super().__init__()
        self.url = url
        self.frame_size = frame_size

        # Keep exact same observation & action spaces so ppo-fnaf-night1.zip transfers directly
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(9)

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 720})
        self.page = self.context.new_page()

        self._load_and_enter_menu()
        self.first_reset = True
        self.current_night = 1
        self.last_episode_won = False

    def _load_and_enter_menu(self):
        self.page.goto(self.url)
        self.page.add_style_tag(
            content="#wrongRotation { display: none !important; } #game { display: block !important; }"
        )

        start = time.time()
        while (time.time() - start) < 60:
            done = self.page.evaluate("""() => {
                const dl = document.getElementById('download-bar');
                const ex = document.getElementById('extract-bar');
                const vis = (el) => el && el.offsetParent !== null && getComputedStyle(el).display !== 'none';
                return !vis(dl) && !vis(ex);
            }""")
            if done:
                break
            self.page.wait_for_timeout(500)

        self.page.wait_for_timeout(9000)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Case A: Just beat Night 1 -> game naturally advances to Night 2 after the 6 AM celebration!
        if self.last_episode_won and self.current_night == 1:
            self.current_night = 2
            self.last_episode_won = False
            print("[Night 2] Night 1 cleared! Waiting 10s for Night 2 transition...", flush=True)
            self.page.wait_for_timeout(10000)
            print("[Night 2] Office live! Training on Night 2...", flush=True)
            self.steps_survived = 0
            return self._get_frame(), {"night": self.current_night}

        # Case B: First run OR died/beat Night 2 -> start fresh from Night 1 (New Game)
        if not self.first_reset:
            self._load_and_enter_menu()
        self.first_reset = False
        self.current_night = 1
        self.last_episode_won = False

        print("[Night 1] Clicking New Game (230, 418) and waiting 5s...", flush=True)
        self.page.mouse.click(230, 418)
        self.page.wait_for_timeout(5000)

        print("[Night 1] Skipping newspaper (640, 360) and waiting 5s...", flush=True)
        self.page.mouse.click(640, 360)
        self.page.wait_for_timeout(5000)

        print("[Night 1] Office live! Training started.", flush=True)
        self.steps_survived = 0
        obs = self._get_frame()
        return obs, {"night": self.current_night}

    def step(self, action):
        if action == 0:
            self.page.wait_for_timeout(250)
        elif action == 1:
            self.page.mouse.move(100, 360)
            self.page.wait_for_timeout(200)
            self.page.mouse.click(48, 472)
        elif action == 2:
            self.page.mouse.move(100, 360)
            self.page.wait_for_timeout(200)
            self.page.mouse.click(48, 576)
        elif action == 3:
            self.page.mouse.move(1180, 360)
            self.page.wait_for_timeout(200)
            self.page.mouse.click(1232, 472)
        elif action == 4:
            self.page.mouse.move(1180, 360)
            self.page.wait_for_timeout(200)
            self.page.mouse.click(1232, 576)
        elif action == 5:
            self.page.mouse.move(530, 668)
            self.page.wait_for_timeout(150)
            self.page.mouse.move(640, 360)
        elif action == 6:
            # CAM 2B (West Hall Corner)
            self.page.mouse.click(985, 625)
        elif action == 7:
            # CAM 4B (East Hall Corner)
            self.page.mouse.click(1090, 625)
        elif action == 8:
            # CAM 1C (Pirate Cove - Foxy's cam for Night 2!)
            self.page.mouse.click(950, 470)

        self.page.wait_for_timeout(150)
        self.steps_survived += 1

        raw_img = self._capture_raw_bgr()
        obs = cv2.resize(raw_img, self.frame_size, interpolation=cv2.INTER_AREA)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

        reward, terminated = self._check_status(raw_img)
        if terminated:
            won = reward >= 90.0
            self.last_episode_won = won
            outcome = "6 AM WIN!" if won else "Jumpscare / Game Over"
            print(
                f"[Night {self.current_night}] Episode ended at step {self.steps_survived} -> "
                f"{outcome} (Reward: {reward})",
                flush=True,
            )

        return obs, reward, terminated, False, {"night": self.current_night, "won": reward >= 90.0}

    def _capture_raw_bgr(self):
        png_bytes = self.page.screenshot()
        nparr = np.frombuffer(png_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _get_frame(self):
        bgr = self._capture_raw_bgr()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, self.frame_size, interpolation=cv2.INTER_AREA)

    def _check_status(self, bgr_img):
        # 1. Check for 6 AM win screen
        if self.steps_survived > 200:
            center_roi = bgr_img[300:420, 520:760]
            bright_center = int(np.sum(np.all(center_roi > 230, axis=-1)))
            corners_dark = int(np.mean(bgr_img[0:100, 0:100])) < 15
            if bright_center > 400 and corners_dark:
                # Extra bonus (+150.0) for beating Night 2!
                win_reward = 150.0 if self.current_night == 2 else 100.0
                return win_reward, True

        # 2. Check for Game Over / Jumpscare static screen
        if self.steps_survived > 35:
            b, g, r = cv2.split(bgr_img)
            color_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))
            mean_brightness = np.mean(bgr_img)
            power_roi = bgr_img[625:665, 35:230]
            clock_roi = bgr_img[25:65, 1130:1250]
            hud_white = int(np.sum(np.all(power_roi > 200, axis=-1))) + int(np.sum(np.all(clock_roi > 200, axis=-1)))

            if hud_white == 0 and color_diff < 4.0 and 40 < mean_brightness < 160:
                return -100.0, True

        return 0.1, False

    def close(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()
