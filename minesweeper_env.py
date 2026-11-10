import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import sync_playwright

class FastWebMinesweeperEnv(gym.Env):
    def __init__(self, board_size=(9, 9)):
        super().__init__()
        self.rows, self.cols = board_size
        
        # 81 discrete tile actions
        self.action_space = spaces.Discrete(self.rows * self.cols)
        
        # 9x9 numerical grid state (-1 = covered/unrevealed, 0-8 = revealed neighbor numbers)
        self.observation_space = spaces.Box(
            low=-1, high=8, shape=(self.rows, self.cols), dtype=np.int32
        )
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 720})
        self.page = self.context.new_page()
        self.page.goto("http://localhost:8000/MinesweeperMania.html")

        # Disable rotation overlay CSS
        self.page.add_style_tag(
            content="#wrongRotation { display: none !important; } #game { display: block !important; }"
        )
        self.page.wait_for_timeout(300)

        # Initial click on main menu center Play button
        box = self.page.locator("#game").bounding_box() or {"x": 0, "y": 0, "width": 1280, "height": 720}
        self.page.mouse.click(box["x"] + box["width"] * 0.50, box["y"] + box["height"] * 0.63)
        self.page.wait_for_timeout(300)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Click smiley face icon at the top center to reset the board instantly
        box = self.page.locator("#game").bounding_box() or {"x": 0, "y": 0, "width": 1280, "height": 720}
        smiley_x = box["x"] + box["width"] * 0.50
        smiley_y = box["y"] + box["height"] * 0.05
        self.page.mouse.click(smiley_x, smiley_y)

        obs = self._get_grid_state()
        return obs, {}

    def step(self, action):
        row = action // self.cols
        col = action % self.cols

        box = self.page.locator("#game").bounding_box() or {"x": 0, "y": 0, "width": 1280, "height": 720}
        grid_top = box["y"] + (box["height"] * 0.10)
        grid_height = box["height"] * 0.78

        cell_w = box["width"] / self.cols
        cell_h = grid_height / self.rows

        click_x = box["x"] + (col + 0.5) * cell_w
        click_y = grid_top + (row + 0.5) * cell_h

        # Fast direct click without artificial delays
        self.page.mouse.click(click_x, click_y)

        obs = self._get_grid_state()
        reward, terminated = self._get_reward_and_status()
        
        return obs, reward, terminated, False, {}

    def _get_grid_state(self):
        # Extract internal grid state matrix via browser JS evaluation
        grid = self.page.evaluate('''() => {
            if (window.aGrid) return window.aGrid;
            return Array(9).fill().map(() => Array(9).fill(-1));
        }''')
        return np.array(grid, dtype=np.int32)

    def _get_reward_and_status(self):
        is_game_over = self.page.evaluate("() => window.gameOver === true")
        is_won = self.page.evaluate("() => window.gameWon === true")

        if is_game_over:
            return -10.0, True
        elif is_won:
            return 100.0, True
        else:
            return 1.0, False

    def close(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()