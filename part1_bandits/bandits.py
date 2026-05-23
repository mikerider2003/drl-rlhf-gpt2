"""
Multi-Armed Bandits — Starter Code
DRL Assignment, Spring 2026, Tilburg University

This file contains:
  - A BanditEnvironment class (k-armed bandit with Gaussian rewards)
  - A complete EpsilonGreedy agent (for reference)
  - Skeleton classes for UCB and ThompsonSampling (TODO: implement these)
  - Experiment runner, regret computation, and plotting utilities

Requirements: numpy, matplotlib
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Environment
# =============================================================================

class BanditEnvironment:
    """A k-armed bandit with Gaussian rewards.

    Each arm i has a true mean reward mu_i drawn from N(0, 1) at initialization.
    Pulling arm i returns a sample from N(mu_i, 1).
    """

    def __init__(self, k=10, seed=None):
        """
        Args:
            k: number of arms
            seed: random seed for reproducibility of the true means
        """
        rng = np.random.RandomState(seed)
        self.k = k
        self.true_means = rng.randn(k)
        self.optimal_arm = int(np.argmax(self.true_means))
        self.optimal_mean = self.true_means[self.optimal_arm]

    def pull(self, arm):
        """Pull an arm and receive a stochastic reward.

        Args:
            arm: integer in [0, k-1]
        Returns:
            reward: sample from N(mu_arm, 1)
        """
        return np.random.randn() + self.true_means[arm]


# =============================================================================
# Agents
# =============================================================================

class EpsilonGreedy:
    """Epsilon-greedy agent (provided as reference)."""

    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)       # number of times each arm was pulled
        self.q_values = np.zeros(k)     # estimated value of each arm

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return int(np.argmax(self.q_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        # Incremental mean update
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class UCB:
    """Upper Confidence Bound (UCB1) agent.

    TODO: Implement the UCB1 algorithm.

    Recall the UCB1 arm selection rule:
        A_t = argmax_a [ Q(a) + c * sqrt( ln(t) / N(a) ) ]

    where Q(a) is the estimated value of arm a, N(a) is the number of times
    arm a has been pulled, t is the current timestep, and c is an exploration
    parameter (typically c = 2 or c = sqrt(2)).

    Hint: Arms that have never been pulled (N(a) = 0) should be selected first
    before applying the UCB formula.
    """

    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c
        self.counts = np.zeros(k)
        self.q_values = np.zeros(k)
        self.t = 0  # total number of steps

    def select_arm(self):
        self.t += 1
        # TODO: implement UCB1 arm selection
        # 1. If any arm has count 0, select it
        # 2. Otherwise, select arm with highest Q(a) + c * sqrt(ln(t) / N(a))
        raise NotImplementedError("TODO: implement UCB1 arm selection")

    def update(self, arm, reward):
        # TODO: update counts and q_values (same incremental mean as EpsilonGreedy)
        raise NotImplementedError("TODO: implement UCB1 update")


class ThompsonSampling:
    """Thompson Sampling agent for Gaussian bandits.

    TODO: Implement Thompson Sampling for Gaussian rewards.

    Thompson Sampling maintains a posterior distribution over each arm's mean
    reward and selects arms by sampling from these posteriors.

    For Gaussian bandits with known variance sigma^2 = 1 and a
    N(0, 1) prior on each arm's mean, the posterior after observing n pulls
    with sample mean x_bar is:

        mu_a | data  ~  N( n * x_bar / (n + 1),  1 / (n + 1) )

    (This comes from conjugate Bayesian updating of a Gaussian prior with
    a Gaussian likelihood.)

    The algorithm:
        1. For each arm a, sample theta_a ~ posterior(a)
        2. Select arm with highest theta_a

    Hint: use np.random.normal(mean, std) to sample from the posterior.
    Note that std = sqrt(variance), so std = sqrt(1 / (n + 1)) = 1/sqrt(n+1).
    """

    def __init__(self, k):
        self.k = k
        self.counts = np.zeros(k)
        self.sum_rewards = np.zeros(k)  # sum of rewards for each arm

    def select_arm(self):
        # TODO: implement Thompson Sampling arm selection
        # 1. For each arm, compute posterior mean and std
        # 2. Sample theta_a from the posterior
        # 3. Select arm with highest theta_a
        raise NotImplementedError("TODO: implement Thompson Sampling arm selection")

    def update(self, arm, reward):
        # TODO: update counts and sum_rewards
        raise NotImplementedError("TODO: implement Thompson Sampling update")


# =============================================================================
# Experiment runner
# =============================================================================

def run_experiment(agent, bandit, n_steps):
    """Run one experiment: agent interacts with bandit for n_steps.

    Args:
        agent: an agent object with select_arm() and update() methods
        bandit: a BanditEnvironment
        n_steps: number of timesteps

    Returns:
        rewards: np.array of shape (n_steps,) — reward at each step
        arms: np.array of shape (n_steps,) — arm selected at each step
    """
    rewards = np.zeros(n_steps)
    arms = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        arms[t] = arm

    return rewards, arms


def compute_cumulative_regret(bandit, arms):
    """Compute cumulative regret over time.

    Regret at step t = (optimal arm's true mean) - (selected arm's true mean).
    Cumulative regret = sum of regrets up to step t.

    Args:
        bandit: BanditEnvironment (to access true means and optimal arm)
        arms: np.array of selected arms at each step

    Returns:
        cumulative_regret: np.array — cumulative regret at each step
    """
    instantaneous_regret = bandit.optimal_mean - bandit.true_means[arms]
    cumulative_regret = np.cumsum(instantaneous_regret)
    return cumulative_regret


def run_multiple_experiments(make_agent_fn, k, n_steps, n_runs, seed_offset=0):
    """Run multiple experiments and average the cumulative regret.

    Args:
        make_agent_fn: callable that takes k and returns a fresh agent
        k: number of arms
        n_steps: timesteps per experiment
        n_runs: number of independent runs (different bandit instances)
        seed_offset: offset for bandit seeds (for reproducibility)

    Returns:
        mean_regret: np.array of shape (n_steps,) — mean cumulative regret
        std_regret: np.array of shape (n_steps,) — std of cumulative regret
    """
    all_regrets = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        bandit = BanditEnvironment(k=k, seed=seed_offset + run)
        agent = make_agent_fn(k)
        _, arms = run_experiment(agent, bandit, n_steps)
        all_regrets[run] = compute_cumulative_regret(bandit, arms)

    mean_regret = np.mean(all_regrets, axis=0)
    std_regret = np.std(all_regrets, axis=0)
    return mean_regret, std_regret


def plot_regret(results, title="Cumulative Regret Comparison"):
    """Plot cumulative regret curves for multiple agents.

    Args:
        results: dict mapping agent_name -> (mean_regret, std_regret)
        title: plot title
    """
    plt.figure(figsize=(10, 6))

    for name, (mean_reg, std_reg) in results.items():
        steps = np.arange(1, len(mean_reg) + 1)
        plt.plot(steps, mean_reg, label=name)
        plt.fill_between(steps,
                         mean_reg - std_reg,
                         mean_reg + std_reg,
                         alpha=0.15)

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regret_comparison.png", dpi=150)
    plt.show()
    print("Plot saved to regret_comparison.png")


# =============================================================================
# Main — run experiments
# =============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    K = 10            # number of arms
    N_STEPS = 1000    # timesteps per run
    N_RUNS = 200      # number of independent runs (averaged for smoother curves)

    results = {}

    # --- Epsilon-Greedy (provided) ---
    print("Running Epsilon-Greedy...")
    mean_reg, std_reg = run_multiple_experiments(
        make_agent_fn=lambda k: EpsilonGreedy(k, epsilon=0.1),
        k=K, n_steps=N_STEPS, n_runs=N_RUNS
    )
    results["Epsilon-Greedy (eps=0.1)"] = (mean_reg, std_reg)

    # --- UCB (TODO) ---
    # Uncomment the following once you have implemented the UCB class:
    #
    # print("Running UCB...")
    # mean_reg, std_reg = run_multiple_experiments(
    #     make_agent_fn=lambda k: UCB(k, c=2.0),
    #     k=K, n_steps=N_STEPS, n_runs=N_RUNS
    # )
    # results["UCB (c=2)"] = (mean_reg, std_reg)

    # --- Thompson Sampling (TODO) ---
    # Uncomment the following once you have implemented ThompsonSampling:
    #
    # print("Running Thompson Sampling...")
    # mean_reg, std_reg = run_multiple_experiments(
    #     make_agent_fn=lambda k: ThompsonSampling(k),
    #     k=K, n_steps=N_STEPS, n_runs=N_RUNS
    # )
    # results["Thompson Sampling"] = (mean_reg, std_reg)

    # --- Plot ---
    plot_regret(results)
