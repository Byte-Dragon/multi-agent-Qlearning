import time

import numpy as np

from multiagent import scenarios
from multiagent.environment import MultiAgentEnv
from utils.util import random_walk
from utils.channel_model import Channel
import utils.config as conf

if __name__ == "__main__":
    # scenario = scenarios.load("simple_urban_comm.py").Scenario(conf.RANDOM_SEED)
    scenario = scenarios.load("simple_urban_comm.py").Scenario()
    # create world
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    channel = Channel(world.policy_agents, world.scripted_agents)

    # # 根据距离来确立连接
    # res_map, res_matrix = channel.link_matrix(by_signal=False)
    # print(f'根据距离来确立连接关系：{res_map}\n连接关系矩阵：\n{res_matrix[:, :, 0]}')
    # print(f'距离值矩阵：\n{res_matrix[:, :, 1]}')
    #
    # # 根据信号强度来确立连接
    # res_map2, res_matrix2 = channel.link_matrix(by_signal=True)
    # # print(f'根据信号强度连接关系：{res_map2}\n关系矩阵：\n{res_matrix2}')
    # print(f'根据信号强度连接关系：{res_map2}\n连接关系矩阵：\n{res_matrix2[:, :, 0]}')
    # print(f'距离值矩阵：\n{res_matrix2[:, :, 1]}')

    # for agent in world.policy_agents:
    #     print(f'agent连接关系：{agent.name}, {agent.link}')
    # res = np.zeros((len(world.policy_agents),len(world.scripted_agents)))
    # for i,agent in enumerate(world.policy_agents):
    #     for j,terminal in enumerate(world.scripted_agents):
    #         res[i][j] = channel.achievable_data_rate(agent, terminal)
    # print(res)
    for i in range(100):
        env.render()
        for agent in world.scripted_agents:
            random_walk(world, agent, step=True)
            break
        time.sleep(0.5)
    # env.render()
    # time.sleep(30)
    # for agent in world.agents:
    #     print(f"{agent.links}_{agent.name}_{agent.state.p_pos}_{agent.color}")
    # for agent in world.landmarks:
    #     print(f"landmarks:\n{agent.links}_{agent.name}_{agent.state.p_pos}_{agent.color}")


