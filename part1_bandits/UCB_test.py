# part1_bandits/UCB_test.py
# python -m part1_bandits.UCB_test

import numpy as np
from part1_bandits.bandits import UCB, BanditEnvironment

def run_one_ucb_cycle():
    np.random.seed(42)

    k = 3
    agent = UCB(k=k, c=2.0)
    bandit = BanditEnvironment(k=k, seed=123)

    print("True means:", bandit.true_means)
    print("Optimal arm:", bandit.optimal_arm)
    print()

    for step in range(10):
        print(f"Step {step}")

        arm = agent.select_arm()
        reward = bandit.pull(arm)

        print("Selected arm:", arm)
        print("Reward:", reward)

        agent.update(arm, reward)

        print("Counts:", agent.counts)
        print("Q-values:", agent.q_values)
        print("-" * 40)

    print("Full cycle finished")


if __name__ == "__main__":
    run_one_ucb_cycle()