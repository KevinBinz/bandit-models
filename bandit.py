from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import operator


class Actions:
    A = 'a'
    B = 'b'
    C = 'c'
    D = 'd'
    E = 'e'
    F = 'f'
    G = 'g'
    H = 'h'
    I = 'i'
    J = 'j'


class BanditArms:
    def __init__(self):
        self.arms = dict()
        self.arms[Actions.A] = norm(-0.4, 2)
        self.arms[Actions.B] = norm(0.5, 2)
        self.arms[Actions.C] = norm(-1.8, 2)
        self.arms[Actions.D] = norm(1.7, 2)
        self.arms[Actions.E] = norm(0.3, 2)
        self.arms[Actions.F] = norm(1.4, 2)
        self.arms[Actions.G] = norm(-1.5, 2)
        self.arms[Actions.H] = norm(-0.1, 2)
        self.arms[Actions.I] = norm(-1.1, 2)
        self.arms[Actions.J] = norm(0.9, 2)

    def get_rewards(self, action, num_samples):
        arm = self.arms[action]
        sample = arm.rvs(size=num_samples)
        return sample

    def get_actions(self):
        return list(self.arms.keys())


class BanditTestbed:
    def __init__(self, bandit, epsilon):
        self.epsilon = epsilon
        self.bandit = bandit
        self.reward_count = dict()
        self.reward_estimate = dict()
        for action in self.bandit.get_actions():
            self.reward_count[action] = 0
            self.reward_estimate[action] = 0

    def choose_action(self):
        random_num = random.uniform(0, 1)
        if random_num > self.epsilon:
            # Exploitation - pursue the argmax action (the one with largest reward_estimate
            argmax_action = max(self.reward_estimate.items(), key=operator.itemgetter(1))[0]
            return argmax_action
        else:
            # Exploration - choose an action at random
            random_action = random.choice(self.bandit.get_actions())
            return random_action

    def update_estimator(self, action, reward):
        new_count = self.reward_count[action] + 1
        action_diff = reward - self.reward_estimate[action]

        self.reward_count[action] = new_count
        self.reward_estimate[action] = self.reward_estimate[action] + (action_diff/new_count)

    def run(self, time_steps):
        for i in range(0, time_steps):
            action = self.choose_action()
            rewards = self.bandit.get_rewards(action=action, num_samples=1)
            self.update_estimator(action, rewards[0])

    def output_estimates(self):
        for action in self.bandit.get_actions():
            actual = self.bandit.arms[action].mean()
            estimate = self.reward_estimate[action]
            diff = abs(actual - estimate)
            print("Action {}. Actual {} Estimate {} Diff {}".format(action, actual, estimate, diff))


if __name__ == "__main__":
    bandit = BanditArms()

    # View Bandit Distribution
    df = pd.DataFrame(columns=["Category", "Values"])
    actions = bandit.get_actions()
    for action in actions:
        num_samples = 30
        sample = bandit.get_rewards(action=action, num_samples=num_samples)
        df_temp = pd.DataFrame({"Category": [action] * num_samples, "Values": sample})
        df = df.append(df_temp, ignore_index=True)

    ax = sns.violinplot(x="Category", y="Values", data=df, inner="stick")
    plt.show(ax)

    # Build a Distribution Estimator
    testbed = BanditTestbed(bandit=bandit, epsilon=0.3)
    testbed.run(time_steps=5000)
    testbed.output_estimates()
