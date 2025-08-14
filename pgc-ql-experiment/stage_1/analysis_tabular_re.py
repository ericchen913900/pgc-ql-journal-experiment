import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

def moving_average(data, window_size):
    """計算移動平均。"""
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def plot_learning_curves(env_name, results_dir, plots_dir):
    """繪製主要效能比較圖。"""
    plt.figure(figsize=(16, 9))
    window_size = 100
    
    agent_files = [f for f in os.listdir(results_dir) if f.startswith(env_name) and f.endswith('.json')]
    
    for agent_file in sorted(agent_files):
        agent_name = agent_file.replace(f'{env_name}_', '').replace('_logs.json', '')
        with open(os.path.join(results_dir, agent_file), 'r') as f:
            logs = json.load(f)
        
        rewards_data = np.array([log['rewards'] for log in logs])
        mean_rewards = np.mean(rewards_data, axis=0)
        
        smoothed_mean = moving_average(mean_rewards, window_size)
        episodes = np.arange(window_size - 1, len(mean_rewards))
        
        if smoothed_mean.any():
            plt.plot(episodes, smoothed_mean, label=agent_name, linewidth=2.5)

    plt.title(f'演算法在「{env_name}」環境下的效能比較', fontsize=20)
    plt.xlabel('回合 (Episodes)', fontsize=14)
    plt.ylabel(f'平均獎勵 (每 {window_size} 回合移動平均)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    if env_name == "AbruptCyclical":
        plt.axvline(x=2000, color='r', linestyle='--', linewidth=2, label='切換至情境B')
        plt.axvline(x=4000, color='g', linestyle='--', linewidth=2, label='切換回情境A')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'performance_{env_name}.png'))
    plt.close()
    print(f"環境 '{env_name}' 的效能圖已儲存。")

def plot_pool_size(env_name, results_dir, plots_dir):
    """繪製 PGC-QL v2.0 的策略池大小變化圖。"""
    agent_file = os.path.join(results_dir, f'{env_name}_PGC-QL (v2.0)_logs.json')
    if not os.path.exists(agent_file): return

    with open(agent_file, 'r') as f:
        logs = json.load(f)
    
    pool_size_data = np.array([[log.get('pool_size', 1) for log in seed_log['extra']] for seed_log in logs])
    mean_pool_size = np.mean(pool_size_data, axis=0)
    
    plt.figure(figsize=(16, 9))
    plt.plot(range(len(mean_pool_size)), mean_pool_size, label='平均策略池大小', color='purple', linewidth=2.5)
    
    plt.title(f'PGC-QL (v2.0) 在「{env_name}」環境下的策略池大小動態', fontsize=20)
    plt.xlabel('回合 (Episodes)', fontsize=14)
    plt.ylabel('平均策略池大小 (K)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    if env_name == "AbruptCyclical":
        plt.axvline(x=2000, color='r', linestyle='--', linewidth=2)
        plt.axvline(x=4000, color='g', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'pool_size_{env_name}.png'))
    plt.close()
    print(f"環境 '{env_name}' 的策略池大小圖已儲存。")

def analyze_and_print_stats(env_name, results_dir):
    """計算並印出統計數據。"""
    print(f"\n--- 環境「{env_name}」的統計分析 ---")
    
    agent_files = [f for f in os.listdir(results_dir) if f.startswith(env_name) and f.endswith('.json')]
    results = {}
    for agent_file in agent_files:
        agent_name = agent_file.replace(f'{env_name}_', '').replace('_logs.json', '')
        with open(os.path.join(results_dir, agent_file), 'r') as f:
            logs = json.load(f)
        rewards_data = np.array([log['rewards'] for log in logs])
        results[agent_name] = rewards_data

    # 累積總獎勵
    print("\n[累積總獎勵 (均值 ± 標準差)]")
    total_rewards = {name: np.sum(data, axis=1) for name, data in results.items()}
    sorted_rewards = sorted(total_rewards.items(), key=lambda item: np.mean(item[1]), reverse=True)
    for name, rewards_arr in sorted_rewards:
        print(f"{name:<15}: {np.mean(rewards_arr):,.2f} ± {np.std(rewards_arr):,.2f}")

    # 與 PGC-QL (v2.0) 進行 t-檢定
    if 'PGC-QL (v2.0)' in total_rewards:
        pgcql_rewards = total_rewards['PGC-QL (v2.0)']
        print("\n[與 PGC-QL (v2.0) 的成對 t-檢定 p-value]")
        for name, rewards_arr in total_rewards.items():
            if name != 'PGC-QL (v2.0)':
                t_stat, p_value = stats.ttest_ind(pgcql_rewards, rewards_arr, equal_var=False)
                print(f"{name:<15}: p = {p_value:.4f} {'(差異顯著)' if p_value < 0.05 else ''}")

def main():
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    results_dir = 'results_re_tabular'
    plots_dir = os.path.join(results_dir, 'plots')
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)

    env_names = sorted(list(set(f.split('_')[0] for f in os.listdir(results_dir) if f.endswith('.json'))))
    if not env_names:
        print(f"在 '{results_dir}' 中找不到結果檔案。請先執行 main_tabular_re.py。")
        return

    for env_name in env_names:
        print(f"\n{'='*50}\n正在分析環境: {env_name}\n{'='*50}")
        plot_learning_curves(env_name, results_dir, plots_dir)
        plot_pool_size(env_name, results_dir, plots_dir)
        analyze_and_print_stats(env_name, results_dir)

if __name__ == '__main__':
    main()
