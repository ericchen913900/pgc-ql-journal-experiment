import numpy as np
import abc
import random

class BaseMaze(abc.ABC):
    """環境的抽象基礎類別，定義通用介面。"""
    def __init__(self, size=15):
        self.size = size
        self.action_space_n = 4  # 0:上, 1:下, 2:左, 3:右
        self.observation_space_shape = (size, size)
        self.agent_pos = None

    @abc.abstractmethod
    def step(self, action):
        """執行一個動作。"""
        pass

    @abc.abstractmethod
    def reset(self):
        """重設環境。"""
        pass
    
    @abc.abstractmethod
    def update_env(self, episode):
        """根據當前回合數更新環境，用於非穩態設計。"""
        pass

class AbruptCyclicalMaze(BaseMaze):
    """環境一：突變與循環。"""
    def __init__(self, size=15):
        super().__init__(size)
        self._contexts_config = {
            'A': {'start': (0, 0), 'goal': (size - 1, size - 1), 'obstacles': {(i, size // 2) for i in range(2, size - 2)}},
            'B': {'start': (size // 2, size // 2), 'goal': (0, 0), 'obstacles': {(size // 2, i) for i in range(2, size - 2)}}
        }
        self.context = None
        self.switch_context('A')

    def switch_context(self, context_name):
        if self.context != context_name:
            self.context = context_name
            config = self._contexts_config[context_name]
            self.start_pos, self.goal_pos, self.obstacles = config['start'], config['goal'], config['obstacles']
            print(f"\n--- [環境一] 已切換至情境 {self.context} ---")

    def update_env(self, episode):
        if episode == 2001: self.switch_context('B')
        elif episode == 4001: self.switch_context('A')

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(self.size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(self.size - 1, col + 1)
        next_pos = (row, col)

        if next_pos in self.obstacles:
            reward, done = -50.0, False
            self.agent_pos = self.agent_pos
        elif next_pos == self.goal_pos:
            reward, done = 100.0, True
            self.agent_pos = next_pos
        else:
            reward, done = -1.0, False
            self.agent_pos = next_pos
        return self.agent_pos, reward, done

class GradualMaze(BaseMaze):
    """環境二：漸變。"""
    def __init__(self, size=15):
        super().__init__(size)
        self.start_pos, self.goal_pos = (0, 0), (size - 1, size - 1)
        self.wall_pos = size // 2
        self.update_obstacles()

    def update_obstacles(self):
        self.obstacles = {(i, self.wall_pos) for i in range(self.size // 2 - 2, self.size // 2 + 3)}

    def update_env(self, episode):
        if 1001 <= episode <= 5000 and episode % 100 == 0:
            new_pos = self.wall_pos + random.choice([-1, 1])
            if 1 < new_pos < self.size - 2:
                self.wall_pos = new_pos
                self.update_obstacles()

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(self.size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(self.size - 1, col + 1)
        next_pos = (row, col)

        if next_pos in self.obstacles:
            reward, done = -50.0, False
            self.agent_pos = self.agent_pos
        elif next_pos == self.goal_pos:
            reward, done = 100.0, True
            self.agent_pos = next_pos
        else:
            reward, done = -1.0, False
            self.agent_pos = next_pos
        return self.agent_pos, reward, done

class RewardChangeMaze(BaseMaze):
    """環境三：獎勵函數變化。"""
    def __init__(self, size=15):
        super().__init__(size)
        self.start_pos, self.goal_pos = (0, 0), (size - 1, size - 1)
        self.obstacles = {(size // 2, i) for i in range(size)}
        self.obstacles.remove((size // 2, size // 2))
        self.reward_mode = None
        self.switch_reward_mode('HighRisk')

    def switch_reward_mode(self, mode_name):
        if self.reward_mode != mode_name:
            self.reward_mode = mode_name
            if mode_name == 'HighRisk':
                self.goal_reward, self.obstacle_penalty = 200.0, -100.0
            elif mode_name == 'LowRisk':
                self.goal_reward, self.obstacle_penalty = 50.0, -5.0
            print(f"\n--- [環境三] 獎勵模式已切換至: {self.reward_mode} ---")

    def update_env(self, episode):
        if episode == 2001:
            self.switch_reward_mode('LowRisk')

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(self.size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(self.size - 1, col + 1)
        next_pos = (row, col)

        if next_pos in self.obstacles:
            reward, done = self.obstacle_penalty, False
            self.agent_pos = self.agent_pos
        elif next_pos == self.goal_pos:
            reward, done = self.goal_reward, True
            self.agent_pos = next_pos
        else:
            reward, done = -1.0, False
            self.agent_pos = next_pos
        return self.agent_pos, reward, done
