import operator
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm


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
        """
        :param action: object of type Actions.
        :param num_samples: how many samples to draw from the arm's distribution.
        :return: A list of size=num_samples
        """
        arm = self.arms[action]
        sample = arm.rvs(size=num_samples)
        return sample

    def get_actions(self):
        return list(self.arms.keys())

    def show_distribution(self, num_samples):
        df = pd.DataFrame(columns=["Category", "Values"])
        actions = self.get_actions()
        for action in actions:
            sample = self.get_rewards(action=action, num_samples=num_samples)
            df_temp = pd.DataFrame({"Category": [action] * num_samples, "Values": sample})
            df = df.append(df_temp, ignore_index=True)

        ax = sns.violinplot(x="Category", y="Values", data=df, inner="stick")
        plt.show(ax)


class BanditTestbed:
    def __init__(self, bandit, epsilon):
        self.epsilon = epsilon
        self.bandit = bandit

        # The following are used for (action-specific) estimation
        self.reward_count = dict()
        self.reward_estimate = dict()
        for action in self.bandit.get_actions():
            self.reward_count[action] = 0
            self.reward_estimate[action] = 0

        # The following are used for global performance visualization
        self.count_correct_action = 0
        self.sum_reward = 0.0
        self.list_count_correct_action = []
        self.list_timesteps = []
        self.list_sum_reward = []

    def run(self, time_steps):
        for i in range(0, time_steps):
            action = self.choose_action()
            reward = self.bandit.get_rewards(action=action, num_samples=1)[0]
            self.update_estimator(action, reward)
            self.update_perf_stats(action, reward, i)

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

    def update_perf_stats(self, action, reward, timestep):
        self.sum_reward = self.sum_reward + reward
        if action == Actions.D:
            self.count_correct_action = self.count_correct_action + 1

        self.list_sum_reward.append(self.sum_reward)
        self.list_count_correct_action.append(self.count_correct_action)
        self.list_timesteps.append(timestep)

    def display_results(self):
        df = pd.DataFrame({"Timestep": self.list_timesteps,
                           "SumReward": self.list_sum_reward,
                           "CountOptimal": self.list_count_correct_action})
        df["PercentOptimal"] = df["CountOptimal"] / (df["Timestep"] + 1)
        df["AverageReward"] = df["SumReward"] / (df["Timestep"] + 1)
        ax = sns.lineplot(x="Timestep", y="PercentOptimal", data=df)
        plt.show(ax)
        ax = sns.lineplot(x="Timestep", y="AverageReward", data=df)
        plt.show(ax)

        for action in self.bandit.get_actions():
            actual = self.bandit.arms[action].mean()
            estimate = self.reward_estimate[action]
            diff = abs(actual - estimate)
            print("{}) Actual vs Estimate: {:.3f} vs {:.3f}. Diff {:.3f}".format(action, actual, estimate, diff))


if __name__ == "__main__":
    bandit = BanditArms()
    bandit.show_distribution(num_samples=30)

    # Build a Distribution Estimator
    testbed = BanditTestbed(bandit=bandit, epsilon=0.3)
    testbed.run(time_steps=5000)
    testbed.display_results()
