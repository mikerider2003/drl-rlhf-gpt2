# Deep Reinforcement Learning Assignment

This repository contains the code, outputs, and report files for the Deep Reinforcement Learning assignment.

## Table of Contents

- [Deep Reinforcement Learning Assignment](#deep-reinforcement-learning-assignment)
  - [Table of Contents](#table-of-contents)
  - [Directory links](#directory-links)
    - [Most Important files](#most-important-files)
    - [Plots](#plots)
    - [Supplementary files](#supplementary-files)
  - [Project tree](#project-tree)

## Directory links

### Most Important files

- [part1_bandits/bandits.py](part1_bandits/bandits.py)  
  Main implementation file for Part 1, containing the bandit environment and the agents.

- [part2_rlhf/rlhf_sentiment_output.ipynb](part2_rlhf/rlhf_sentiment_output.ipynb)  
  Executed RLHF notebook for Part 2.

### Plots

- [part1_bandits/regret_comparison.png](part1_bandits/regret_comparison.png)  
  Regret comparison plot for Epsilon-Greedy, UCB, and Thompson Sampling.

- [part2_rlhf/rlhf_results/07_training_curves_kl0.2.png](part2_rlhf/rlhf_results/07_training_curves_kl0.2.png)  
  Training curves for the baseline RLHF run with KL coefficient `β = 0.2`.

- [part2_rlhf/rlhf_results/09_kl_curves_sweep.png](part2_rlhf/rlhf_results/09_kl_curves_sweep.png)  
  KL divergence curves for the different KL coefficient experiments.

- [part2_rlhf/rlhf_results/09_reward_curves_kl_sweep.png](part2_rlhf/rlhf_results/09_reward_curves_kl_sweep.png)  
  Reward curves for the different KL coefficient experiments.

- [part2_rlhf/rlhf_results/09_reward_kl_frontier.png](part2_rlhf/rlhf_results/09_reward_kl_frontier.png)  
  Reward-KL trade-off plot showing the relation between final reward and KL divergence.

### Supplementary files

- [outputs/](outputs/)  
  Contains the `.out` and `.err` files from the submitted cluster/job runs.

- [part1_bandits/](part1_bandits/)  
  Contains the Part 1 tests and the regret comparison plot.

- [part2_rlhf/rlhf_results/](part2_rlhf/rlhf_results/)  
  Contains all saved RLHF outputs, including training logs, generated samples, trained model files, and result plots.

- [part2_rlhf/rlhf_results/06_trained_model_kl0.2/](part2_rlhf/rlhf_results/06_trained_model_kl0.2/)  
  Contains the trained policy model from the baseline RLHF run with KL coefficient `β = 0.2`.

- [part2_rlhf/rlhf_results/08_individual_kl_results/](part2_rlhf/rlhf_results/08_individual_kl_results/)  
  Contains saved result files for the individual KL coefficient experiments used in the KL sweep.

## Project tree

```text
.
├── drl_assignment_spring26.pdf
├── outputs/
├── part1_bandits/
│   ├── Thompson_test.py
│   ├── UCB_test.py
│   ├── bandits.py
│   └── regret_comparison.png
├── part2_rlhf/
│   ├── rlhf_results/
│   │   ├── 06_trained_model_kl0.2/
│   │   ├── 06_training_logs_kl0.2.pkl
│   │   ├── 07_before_after_samples_kl0.2.pkl
│   │   ├── 07_training_curves_kl0.2.png
│   │   ├── 08_individual_kl_results/
│   │   │   ├── kl0.0.pkl
│   │   │   ├── kl0.01.pkl
│   │   │   ├── kl0.05.pkl
│   │   │   ├── kl0.2.pkl
│   │   │   └── kl1.0.pkl
│   │   ├── 09_kl_curves_sweep.png
│   │   ├── 09_reward_curves_kl_sweep.png
│   │   └── 09_reward_kl_frontier.png
│   ├── rlhf_sentiment.ipynb
│   └── rlhf_sentiment_output.ipynb
├── requirements.txt
└── run.sh
```
