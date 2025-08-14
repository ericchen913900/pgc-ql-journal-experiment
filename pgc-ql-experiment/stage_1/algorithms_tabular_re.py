import numpy as np
import abc
import random
from collections import deque

class BaseAgent(abc.ABC):
    """抽象基礎代理類別。"""
    def __init__(self, action_space_n, state_space_shape, **kwargs):
        self.action_space_n = action_space_n
        self.state_space_shape = state_space_shape
    @abc.abstractmethod
    def choose_action(self, state): pass
    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, done): pass
    def get_extra_logs(self): return {}
    def manage_pool(self): pass # 用於 PGC-QL v2.0

# --- 基礎比較基準 ---
class StandardQL(BaseAgent):
    # (此處程式碼與前次相同，為簡潔省略)
    def __init__(self, action_space_n, state_space_shape, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9999, min_epsilon=0.01, **kwargs):
        super().__init__(action_space_n, state_space_shape)
        self.q_table = np.zeros(state_space_shape + (action_space_n,))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.epsilon_decay, self.min_epsilon = epsilon_decay, min_epsilon
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon: return np.random.choice(self.action_space_n)
        else: return np.argmax(self.q_table[state])
    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state]) if not done else 0
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

# --- 新增基準 ---
class ExperienceReplayQL(StandardQL):
    """新增基準：帶有經驗回放的Q-Learning。"""
    def __init__(self, action_space_n, state_space_shape, replay_buffer_size=10000, batch_size=64, **kwargs):
        super().__init__(action_space_n, state_space_shape, **kwargs)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

    def learn(self, state, action, reward, next_state, done):
        # 正常學習一次
        super().learn(state, action, reward, next_state, done)
        
        # 儲存經驗
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # 如果緩衝區足夠大，則進行回放學習
        if len(self.replay_buffer) > self.batch_size:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            for s, a, r, ns, d in minibatch:
                # 使用回放的數據進行額外學習
                super().learn(s, a, r, ns, d)

# --- PGC-QL v2.0 ---
class PGCQL_v2(BaseAgent):
    """PGC-QL v2.0，整合了策略池管理機制。"""
    def __init__(self, action_space_n, state_space_shape, alpha=0.1, gamma=0.99, epsilon=0.1, 
                 theta_new=25.0, ma_alpha=0.05, 
                 phi_prune=0.01, phi_merge=0.95, **kwargs):
        super().__init__(action_space_n, state_space_shape)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.theta_new, self.ma_alpha = theta_new, ma_alpha
        self.phi_prune, self.phi_merge = phi_prune, phi_merge
        
        self.policy_pool = []
        self.policy_usage_counts = []
        self._create_new_policy()
        self.active_policy_index = 0

    def _create_new_policy(self):
        new_policy = {
            'q_table': np.zeros(self.state_space_shape + (self.action_space_n,)),
            'td_error_ma': 0.0,
        }
        self.policy_pool.append(new_policy)
        self.policy_usage_counts.append(0)
        print(f"*** PGCQL-v2: 已創建新策略，當前策略池大小: {len(self.policy_pool)} ***")

    def choose_action(self, state):
        td_errors_ma = [p['td_error_ma'] for p in self.policy_pool]
        best_policy_index = np.argmin(td_errors_ma)
        
        if self.policy_pool[best_policy_index]['td_error_ma'] > self.theta_new and len(self.policy_pool) < 10:
            self._create_new_policy()
            self.active_policy_index = len(self.policy_pool) - 1
        else:
            self.active_policy_index = best_policy_index
        
        self.policy_usage_counts[self.active_policy_index] += 1
        
        active_q_table = self.policy_pool[self.active_policy_index]['q_table']
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space_n)
        else:
            return np.argmax(active_q_table[state])

    def learn(self, state, action, reward, next_state, done):
        active_policy = self.policy_pool[self.active_policy_index]
        q_table = active_policy['q_table']
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state]) if not done else 0
        td_error = reward + self.gamma * next_max - old_value
        q_table[state][action] = old_value + self.alpha * td_error
        current_ma = active_policy['td_error_ma']
        active_policy['td_error_ma'] = (1 - self.ma_alpha) * current_ma + self.ma_alpha * abs(td_error)

    def manage_pool(self):
        """執行策略池管理：合併與淘汰。"""
        if len(self.policy_pool) <= 1:
            return

        # 1. 淘汰 (Pruning)
        total_usage = sum(self.policy_usage_counts)
        if total_usage > 0:
            usage_freq = np.array(self.policy_usage_counts) / total_usage
            # 我們保留至少一個策略，所以只考慮淘汰非最優的策略
            indices_to_prune = np.where(usage_freq < self.phi_prune)[0]
            # 不能淘汰當前最活躍的策略
            most_active = np.argmax(usage_freq)
            indices_to_prune = [i for i in indices_to_prune if i != most_active]
            
            if indices_to_prune:
                # 從後往前刪除以避免索引問題
                for i in sorted(indices_to_prune, reverse=True):
                    if len(self.policy_pool) > 1:
                        del self.policy_pool[i]
                        del self.policy_usage_counts[i]
                        print(f"--- PGCQL-v2: 已淘汰策略 {i} ---")

        # 2. 合併 (Merging) - 僅作簡化演示
        # 一個完整的合併需要更複雜的邏輯來處理相似度計算和Q值平均
        # 此處僅為框架佔位
        
        # 重設使用計數
        self.policy_usage_counts = [0] * len(self.policy_pool)

    def get_extra_logs(self):
        return {'pool_size': len(self.policy_pool)}
