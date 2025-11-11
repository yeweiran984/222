"""
@author: Wenyu Li

* the gymnasium game 2048 environment
"""

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Game2048Env(gym.Env):
    """
    ## Description:

    The 2048 game environment implemented in Gymnasium. The tiles can be moved
    in left, right, up, and down four directions. When two tiles with the same 
    number touch, they merge into one tile with the sum of the two numbers.
    The player wins the game when the largest tile reaches the target.

    ## Installation:

    ```bash
    > pip install .
    ```

    ## Observation Space:

    The observation shape is `(size, size)`, each element is an integer of 
    2^N representing the current state of in the grid.

    ## Action Space:

    The action shape is `(1, )` in the range `{0, 3}` indicationg which 
    direction to move the player.

    - 0: Move left
    - 1: Move right
    - 2: Move up
    - 3: Move down

    ## Reward:

    If new tile is added to the board, the reward is the sum of the values and
    the added values. Otherwise, the reward is -1.
    If any tile reach the target, the reward is +1000.

    ## Starting State

    The starting state is initialized randomly with two tiles placed at random 
    locations on the board. Each tile is placed with a value of 2 or 4 with 0.9 
    and 0.1 probability.

    ## Episode End

    The episode ends when the largest tile reaches target or any other action 
    can player wins or loses the game.

    ## Arguments

    Game 2048 has three parameters for `gymnasium.make` with `render_mode`, 
    `size`, and `target`.
    On reset, the `options` argument is ignored.

    ```python
    >>> import gymnasium as gym
    >>> import game2048
    >>> env = gym.make("Game2048-v0", render_mode="human")
    >>> env
    <OrderEnforcing<PassiveEnvChecker<Game2048Env<Game2048-v0>>>>
    ```

    ## Version History

    *v0: Initial versions release
    """

    metadata = {
        'render_modes': ['human', 'ansi'], 
        'render_fps': 4,
    }

    def __init__(self, render_mode=None, size=4, target=2048):
        self.size = size    # The size of the square grid
        self.target = target    # The target score to reach

        self.window_size = 512  # The size of the window to render the game in pixels

        self.observation_space = spaces.Box(low=0, high=2048, shape=(size, size), dtype=np.int32) # The observation space is a 4x4 grid of integers representing the current state of the game

        self.action_space = spaces.Discrete(4) # The action space is a discrete space with 4 possible actions: right, left, up, down

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros(shape=(self.size, self.size), dtype=np.int32)
        self._add_random_tile()
        self._add_random_tile()

        observation = self.grid.copy()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        if self.render_mode == 'ansi':
            print(self.grid)

        return observation, info
    

    def _get_info(self):
        return {}
    

    def step(self, action):
        prev_grid = self.grid.copy()
        if action == Action.LEFT.value:
            self.grid = self._move_left()
        elif action == Action.RIGHT.value:
            self.grid = self._move_right()
        elif action == Action.UP.value:
            self.grid = self._move_up()
        elif action == Action.DOWN.value:
            self.grid = self._move_down()
        
        if not np.array_equal(prev_grid, self.grid):
            self._add_random_tile()

        terminated = self._check_game_over()
        reward = self._calculate_reward(prev_grid)

        truncated = False

        observation = self.grid.copy()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        if self.render_mode == 'ansi':
            print(self.grid)

        return observation, reward, terminated, truncated, info
    

    def render(self):
        if self.render_mode == 'human':
            print(self.grid)
            return self.grid.copy()
        
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
        if self.render_mode == 'ansi':
            print(self.grid)
        

    def close(self):
        if self.window is not None:
            time.sleep(3)
            pygame.display.quit()
            pygame.quit()


    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((205, 193, 179))
        pix_square_size = self.window_size // self.size

        # 画网格
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (187, 173, 160),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=10,
            )
            pygame.draw.line(
                canvas,
                (187, 173, 160),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=10,
            )

        font_color = [(119, 110, 101), (249, 246, 242)]
        tile_color = [(238, 228, 218), (237, 224, 200), (242, 177, 121), 
                      (245, 149, 99), (246, 124, 95), (246, 94, 59),
                      (237, 207, 114), (237, 204, 98), (237, 200, 80),
                      (237, 197, 63), (237, 194, 46)]
        # 画数字
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid[x, y]
                if tile > 0:
                    if tile <= 2048:
                        bcolor = tile_color[int(np.log2(tile) - 1)]
                    else:
                        bcolor = (0, 0, 0)

                    pygame.draw.rect(
                        canvas, 
                        bcolor, 
                        (pix_square_size * y + 5, pix_square_size * x + 5, pix_square_size - 10, pix_square_size - 10)
                    )

                    if tile >= 100:
                        fsize = 48
                        if tile >= 1000:
                            fsize = 36
                            if tile >= 10000:
                                fsize = 24
                    else:
                        fsize = 64

                    text = str(tile)
                    font = pygame.font.SysFont("microsoftyahei", fsize, bold=True)
                    # font = pygame.font.SysFont("arial", 64)
                    if tile <= 4:
                        fcolor = font_color[0]
                    else:
                        fcolor = font_color[1]
                    text_surface = font.render(text, True, fcolor)
                    text_rect = text_surface.get_rect()
                    text_rect.center = (pix_square_size * (y + 0.5), pix_square_size * (x + 0.5))
                    canvas.blit(text_surface, text_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def _move_left(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            row = self.grid[i]
            non_zero = row[row != 0]
            merged, _ = self._merge(non_zero)

            new_grid[i, :len(merged)] = merged
        return new_grid
    

    def _move_right(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            row = self.grid[i]
            non_zero = row[::-1][row[::-1] != 0]
            merged, _ = self._merge(non_zero)

            new_grid[i, self.size - len(merged):] = merged[::-1]
        return new_grid
    

    def _move_up(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            col = self.grid[:, i]
            non_zero = col[col != 0]
            merged, _ = self._merge(non_zero)

            new_grid[:len(merged), i] = merged
        return new_grid
    

    def _move_down(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            col = self.grid[:, i]
            non_zero = col[::-1][col[::-1] != 0]
            merged, _ = self._merge(non_zero)

            new_grid[self.size - len(merged):, i] = merged[::-1]
        return new_grid
    
    
    def _merge(self, row):
        merged = []
        skip = False
        for j in range(len(row)):
            if skip:
                skip = False
                continue
            if j + 1 < len(row) and row[j] == row[j + 1]:
                merged.append(row[j] * 2)
                skip = True
            else:
                merged.append(row[j])
        return np.array(merged), len(merged) != len(row)


    def _calculate_reward(self, prev_grid):
        reward = np.sum(self.grid)
        return float(reward)


    def _check_game_over(self):
        if np.any(self.grid == 0):
            return False
        for i in range(self.size):
            for j in range(self.size):
                if (j < self.size - 1 and self.grid[i][j] == self.grid[i][j + 1]) or \
                   (i < self.size - 1 and self.grid[i][j] == self.grid[i + 1][j]):
                    return False
        return True
    

    def _add_random_tile(self):
        empty_pos = np.argwhere(self.grid == 0)
        if empty_pos.size > 0:
            r, c = empty_pos[np.random.randint(0, len(empty_pos))]
            self.grid[r, c] = 2 if np.random.random() < 0.9 else 4