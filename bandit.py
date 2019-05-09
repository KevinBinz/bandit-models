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


class BanditArm:
    def __init__(self, reward_distrib):
        self.distrib = reward_distrib

    def get_reward(self, size=1):
        sample = self.distrib.rvs(size=size)
        return sample


class BanditTestbed:
    def __init__(self):
        self.arms = dict()
        self.arms[Actions.A] = BanditArm(norm(-0.4, 2))
        self.arms[Actions.B] = BanditArm(norm(0.5, 2))
        self.arms[Actions.C] = BanditArm(norm(-0.8, 2))
        self.arms[Actions.D] = BanditArm(norm(1.7, 2))
        self.arms[Actions.E] = BanditArm(norm(0.3, 2))
        self.arms[Actions.F] = BanditArm(norm(1.4, 2))
        self.arms[Actions.G] = BanditArm(norm(-1.5, 2))
        self.arms[Actions.H] = BanditArm(norm(-0.1, 2))
        self.arms[Actions.I] = BanditArm(norm(-1.1, 2))
        self.arms[Actions.J] = BanditArm(norm(0.9, 2))

    def get_reward(self, action):
        arm = self.arms[action]
        return arm.get_reward()


if __name__ == "__main__":
    testbed = BanditTestbed()
    for i in range(0, 20):
        print(testbed.get_reward(Actions.A))
