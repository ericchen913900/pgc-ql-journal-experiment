import os
import numpy as np
import json
from tqdm import tqdm

from environments_re import AbruptCyclicalMaze, GradualMaze, RewardChangeMaze
from algorithms_tabular_re import StandardQL, ExperienceReplayQL, PGCQL_v2

def run_single_experiment(env_class, agent_class, agent_params, total_episodes, seed):
    np.random.seed(seed)
    env = env_class()
    agent = agent_class(
        action_space_n=env.action_space_n,
        state_space_shape=env.observation_space_shape,
        **agent_params
    )
    
    logs = {'rewards': [], 'extra': []}
    MANAGE_POOL_EVERY = 500 # 每500回合管理一次策略池
    
    for episode in range(total_episodes):
        env.update_env(episode)
        
        # 週期性地管理策略池
        if episode > 0 and episode % MANAGE_POOL_EVERY == 0:
            agent.manage_pool()

        state = env.reset()
        done = False
        total_reward = 0
        max_steps = env.size * env.size * 2
        
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        
        logs['rewards'].append(total_reward)
        logs['extra'].append(agent.get_extra_logs())
        
    return logs

def main():
    NUM_SEEDS = 50 # 強化統計嚴謹性
    TOTAL_EPISODES = 6000
    
    ENVIRONMENTS = {
        "AbruptCyclical": AbruptCyclicalMaze,
        "Gradual": GradualMaze,
        "RewardChange": RewardChangeMaze
    }

    AGENTS_CONFIG = {
        'StandardQL': {'class': StandardQL, 'params': {}},
        'ER-QL': {'class': ExperienceReplayQL, 'params': {}},
        'PGC-QL (v2.0)': {'class': PGCQL_v2, 'params': {}},
    }

    results_dir = 'results_re_tabular'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for env_name, env_class in ENVIRONMENTS.items():
        print(f"\n{'#'*50}\n#####   開始執行環境: {env_name}   #####\n{'#'*50}")
        for agent_name, config in AGENTS_CONFIG.items():
            print(f"\n===== 正在執行演算法: {agent_name} =====")
            all_seeds_logs = []
            for seed in tqdm(range(NUM_SEEDS), desc=f"{agent_name} on {env_name}"):
                logs = run_single_experiment(
                    env_class, config['class'], config['params'], TOTAL_EPISODES, seed
                )
                all_seeds_logs.append(logs)
            
            filename = f"{env_name}_{agent_name}_logs.json"
            with open(os.path.join(results_dir, filename), 'w') as f:
                json.dump(all_seeds_logs, f)

    print(f"\n所有表格實驗執行完畢！結果已儲存至 '{results_dir}' 資料夾。")

if __name__ == '__main__':
    main()
