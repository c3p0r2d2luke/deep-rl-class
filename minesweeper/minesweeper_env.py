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
        
        # 9x9 numerical grid state (-1 = covered, 0-8 = revealed counts)
        self.observation_space = spaces.Box(
            low=-1, high=8, shape=(self.rows, self.cols), dtype=np.int32
        )
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 720})
        self.page = self.context.new_page()
        self.page.goto("http://localhost:8000/MinesweeperMania.html")

        # Force hide rotation lock overlay
        self.page.add_style_tag(
            content="#wrongRotation { display: none !important; } #game { display: block !important; }"
        )
        # Wait for splash screen to finish loading
        self.page.wait_for_timeout(3500)
        self.game_started = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.game_started:
            # First episode: click exact Play button coordinates (690, 690)
            self.page.mouse.click(690, 690)
            self.page.wait_for_timeout(500)
            self.game_started = True
        else:
            # Subsequent episodes: click the top smiley face restart button at (640, 44)
            self.page.mouse.click(640, 44)
            self.page.wait_for_timeout(150)

        obs, _, _ = self._get_game_state()
        self.last_covered = int(np.sum(obs == -1))
        return obs, {}

    def step(self, action):
        row = action // self.cols
        col = action % self.cols

        # Exact pixel bounds of the centered 9x9 grid on a 1280x720 viewport
        grid_left = 394.0
        grid_top = 92.0
        grid_width = 492.0
        grid_height = 492.0

        cell_w = grid_width / self.cols
        cell_h = grid_height / self.rows

        click_x = grid_left + (col + 0.5) * cell_w
        click_y = grid_top + (row + 0.5) * cell_h

        self.page.mouse.click(click_x, click_y)

        obs, hit_bomb, is_won = self._get_game_state()
        current_covered = int(np.sum(obs == -1))

        if hit_bomb:
            reward = -10.0
            terminated = True
        elif is_won:
            reward = 100.0
            terminated = True
        else:
            # Reward proportional to new tiles uncovered; small penalty if clicking an already-open tile
            newly_revealed = self.last_covered - current_covered
            if newly_revealed > 0:
                reward = float(newly_revealed)
            else:
                reward = -0.5
            terminated = False

        self.last_covered = current_covered
        return obs, reward, terminated, False, {}

    def _get_game_state(self):
        state = self.page.evaluate("""() => {
            const fields = window.imgGameFields || [];
            const bombsTotal = window.bombsInGame || 8;
            let hitBomb = false;
            let coveredCount = 0;
            const flat = [];

            for (let i = 0; i < 81; i++) {
                const f = fields[i];
                if (!f) {
                    flat.push(-1);
                    coveredCount++;
                    continue;
                }
                const name = f.frameName || '';
                if (name.includes('bomb')) {
                    hitBomb = true;
                    flat.push(-1);
                } else if (name.includes('bt_on')) {
                    flat.push(-1);
                    coveredCount++;
                } else if (name.includes('bt_off')) {
                    flat.push(0);
                } else {
                    const match = name.match(/_(\\d+)\\.png$/);
                    flat.push(match ? parseInt(match, 10) : 0);
                }
            }

            const isWon = !hitBomb && (coveredCount === bombsTotal);
            return { flat, hitBomb, isWon };
        }""")

        grid = np.array(state["flat"], dtype=np.int32).reshape((self.rows, self.cols))
        return grid, state["hitBomb"], state["isWon"]


    def close(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()
