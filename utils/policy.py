"""
在本文件中写动作选择的策略，返回包含n个agent的action的action_n
"""
import numpy as np
import random
import utils.config as conf

# individual agent policy
from utils.make_env import make_env


class Policy(object):
    def __init__(self):
        pass

    def action(self, obs, eposide_index):
        raise NotImplementedError()


# decayed-epsilon greedy policy to choose a action
class GreedyPolicy(Policy):
    def __init__(self, env, agent_index, q_table):
        self.env = env
        self.agent = self.env.world.policy_agents[agent_index]
        # print(f"传入了agent: {agent_index}, name: {self.agent.name}")
        self.q_table = q_table
        # print(f'初始化：{self.q_table}')

    def action(self, obs, eposide_index):
        # print(f'传入的观察数组：{obs}.\ntupled: {tuple(obs)}')
        obs = obs[:self.env.world.dim_p]
        # print(f'传入的观察数组：{obs}.\ntupled: {tuple(obs)}')
        obs = tuple(obs)

        # 当前的轮次，用于计算epsilon
        index = eposide_index
        self.check_state_exist(obs)  # 调用这个函数的作用是检查Q值表中有无该状态，如果没有就向表中追加
        # 选择动作的索引
        # 下面的操作就是有epsilon_of_n_tau(index)的概率按Q值表选择最优的，有1-epsilon的概率随机选择动作
        # 随机选动作的意义就是去探索那些可能存在的之前没有发现但是更好的方案/动作/路径
        if np.random.uniform() < epsilon_of_n_tau(index):
            # 随机选择一个动作
            action_index = random.randint(0, conf.NUM_ACTIONS-1)
            # print(f'if random choose action index: {action_index}')
        else:
            # 选择最佳动作（Q值最大的动作）
            state_action = self.q_table[obs]
            # 如果几个动作的Q值同为最大值，从中选一个
            action_index = np.random.choice(np.where(state_action == np.max(state_action))[0])
            # print(f'else argmax state_action: {state_action}, action index: {action_index}')

        # 组织为动作空间的数组
        action = np.zeros(conf.NUM_ACTIONS)
        action[action_index] += conf.DELTA_FORCE
        c = [self.env.get_reward(self.agent)]

        return np.concatenate([action, c]), action_index

    # 检查Q值表中有无状态state，如果没有就向表中追加
    def check_state_exist(self, state):
        state = tuple(state)
        if state not in self.q_table:
            # 向Q表中追加一个状态
            new_row = {state: np.zeros(conf.NUM_ACTIONS)}
            self.q_table.update(new_row)


# 根据公式计算e-greedy policy 参数 epsilon
def epsilon_of_n_tau(n_tau):
    # n_tau 是当前的episode index
    epsilon = conf.EPSILON * (1 - conf.EPSILON) ** (n_tau / (conf.EPSILON * (conf.NUM_ACTIONS ** 2)))
    return epsilon


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 循环1-100，传入n_tau
    n_tau_values = range(1, 101)
    epsilon_values = [epsilon_of_n_tau(n_tau) for n_tau in n_tau_values]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(n_tau_values, epsilon_values)
    plt.title('Epsilon-Greedy Policy: Epsilon vs. Episode Index')
    plt.xlabel('Episode Index (n_tau)')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.show()
