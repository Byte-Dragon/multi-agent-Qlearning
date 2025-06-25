import os, sys
import time
from utils.policy import GreedyPolicy
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import utils.config as conf
import numpy as np

from utils.space_discretizer import SpaceDiscretizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))

if __name__ == '__main__':
    # create env
    scenario = scenarios.load('simple_urban_comm.py').Scenario()
    world = scenario.make_world()  # 这里也调用了一次reset
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.info,
                        shared_viewer=True, discrete_action_space=True)

    # initiate parameters
    q_table = [{} for i in range(env.n)]
    policies = [GreedyPolicy(env, i, q_table[i]) for i in range(env.n)]
    reward_list = []  # 记录所有的accumulated reward，方便绘图
    for eps in range(conf.EPISODES):
        rew_all = [0, 0, 0]  # 记录本轮的accumulated reward
        episode_time = time.time()  # 用于记录运行时间，我们可以通过比较运行时间判断算法效率。

        # 每一轮，重置环境
        obs_n = env.reset()
        for step in range(conf.STEPS):
            # 同一轮训练中的每一个更新step
            if conf.RENDER:
                env.render()
                time.sleep(0.01)
            # 先绘制之前的状态
            # 获取每个agent当前所选择的动作，base on e-greedy policy
            act_n = []
            act_index_n = []
            for i, policy in enumerate(policies):
                # 离散化
                # print(f'step: {step}, origin:{obs_n[i][:env.world.dim_p]}, dis:{discretizer.discretize_point(obs_n[i][:env.world.dim_p])}')
                obs_n[i][:env.world.dim_p] = conf.DISCRETIEZER.discretize_point(obs_n[i][:env.world.dim_p])
                act_i, act_index_i = policy.action(obs_n[i], eps)
                act_n.append(act_i)
                act_index_n.append(act_index_i)

            # 执行动作
            obs_n1, reward_n, done_n, info_n = env.step(act_n)
            # 离散化
            obs_n1[i][:env.world.dim_p] = conf.DISCRETIEZER.discretize_point(obs_n1[i][:env.world.dim_p])
            # update Q-function based on the reward for all agents
            for i in range(len(obs_n)):
                # i 代表的是agent i
                # 为了简便，先重命名相关参数
                s = tuple(obs_n[i][:env.world.dim_p])
                a = act_index_n[i]
                r = reward_n[i]
                s1 = tuple(obs_n1[i][:env.world.dim_p])
                alpha = conf.LEARNING_RATE  # 学习率
                gamma = conf.DISCOUNT_FACTOR  # 折扣率

                # 更新
                policies[i].check_state_exist(s1)  # 需要先验证是否存在新的状态，如果不存在，则新增并初始化
                q_table[i][s][a] = (1 - alpha) * q_table[i][s][a] + alpha * (r + gamma * np.max(q_table[i][s1][:]))
                rew_all[i] += r  # rAll累加当前的收获。
                # rendering
            # end for agent i

            obs_n = obs_n1  # 把下一状态赋值给obs_n，准备开始下一步。

            # set actions for scripted agents
            for agent in env.world.scripted_agents:
                agent.action_callback(env.world, agent, step=True)
            import utils.util as util
            util.update_agent_link(world)
            # print(f"running: [{eps}/{conf.EPISODES}], step: {step}/{conf.STEPS} ")
        # end for step
        reward_list.append(rew_all)
        print(f"Episode [{eps}/{conf.EPISODES}] sum reward: {rew_all} took: {time.time() - episode_time} ")
    # end for episode
    import matplotlib.pyplot as plt

    # 创建图表
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
    line_styles = ['-', '--', '-.', ':']  # 线型列表
    x = list(range(conf.EPISODES))
    for i in range(conf.NUM_LANDMARKS):
        # 自定义线条样式和颜色
        # 这里我们使用循环来生成不同的颜色
        y = list([rew[i] for rew in reward_list])
        color = colors[i]
        linestyle = line_styles[i]
        plt.plot(x, y, color=color, linestyle=linestyle, label=f'Agent-{i}')
        # plt.plot(x, y, color=color, linestyle=linestyle, label=f'Agent-{i}', marker='o')
        # 确保颜色和线型的数量至少与终端数量一致
        # 设置图例
        plt.legend()
        # 设置X轴和Y轴标签
        plt.xlabel('episode')  # X轴标签
        plt.ylabel('Reward')  # Y轴标签
        # 设置图表标题
        # plt.title('Line Graph Example')
        # 显示图表
        plt.show()

    # print("Final Q-Table Values:/n %s" % q_table)
    for i in range(conf.NUM_LANDMARKS):
        np.save(f'q_table_agent_{i}.npy', q_table[i])
