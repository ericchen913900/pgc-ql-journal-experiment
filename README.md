PGC-QL: A Framework for Reinforcement Learning in Non-Stationary Environments
This repository contains the official experimental framework for the research paper on Policy-Gated Contextual Q-Learning (PGC-QL). The project is designed to reproduce, analyze, and extend the experiments intended for submission to top-tier academic journals such as IEEE Transactions on Neural Networks and Learning Systems (TNNLS).

About The Project
Policy-Gated Contextual Q-Learning (PGC-QL) is a novel reinforcement learning framework designed to operate effectively in non-stationary environments. Its core idea is to maintain a dynamic pool of specialized policies and use a lightweight "gating mechanism" to intelligently select, reuse, or create new policies in response to environmental changes. This architecture allows the agent to avoid "catastrophic forgetting," a critical challenge in continual learning.

File Architecture
The experimental setup follows a "Two-Track Validation" architecture to systematically evaluate PGC-QL's core mechanisms and its scalability to high-dimensional spaces.

pgc-ql-journal-experiment/
│
├── 軌道一_表格環境驗證/
│   ├── environments_re.py
│   ├── algorithms_tabular_re.py
│   ├── main_tabular_re.py
│   └── analysis_tabular_re.py
│
├── 軌道二_深度學習驗證/
│   ├── deep_rl_re.py
│   └── meta_rl_re.py
│
├── results_re_tabular/
│   └── plots/
│
└── results_re_deep/
    └── plots/

Getting Started
Prerequisites
Python 3.9+

NumPy, Matplotlib, TQDM

(For Track 2) PyTorch, Gymnasium, Gymnasium[atari], Gymnasium[mujoco]

pip install numpy matplotlib tqdm
# For Track 2, additional libraries are required:
# pip install torch gymnasium[atari] gymnasium[mujoco] autorom-accept-rom-license

How to Run Experiments
Track 1: Core Mechanism Validation (Tabular)
This track validates the core PGC-QL v2.0 mechanisms in discrete grid-world environments.

Run Experiments:

cd 軌道一_表格環境驗證
python main_tabular_re.py

Results will be saved to results_re_tabular/.

Analyze Results:

python analysis_tabular_re.py

Plots will be saved to results_re_tabular/plots/.

Track 2: Deep RL Generalization (Conceptual Framework)
This track provides the conceptual framework and simulation code for testing PGC-QL's scalability.

Run Continual Atari Simulation:

cd 軌道二_深度學習驗證
python deep_rl_re.py

Run Meta-RL Simulation:

python meta_rl_re.py

Core Algorithm Files
algorithms_tabular_re.py: Contains the full implementation of PGC-QL v2.0, which includes the policy pool management mechanisms (pruning and merging).

deep_rl_re.py: Contains the PyTorch model architecture for Deep PGC-QL, demonstrating its extension to high-dimensional visual inputs.