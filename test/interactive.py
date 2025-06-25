#!/usr/test/env python
import os,sys
import time

import numpy as np
import pandas as pd

from utils.policy import GreedyPolicy

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from multiagent.environment import MultiAgentEnv
from policy import InteractivePolicy
import multiagent.scenarios as scenarios
import utils.config as conf

if __name__ == '__main__':
    # parse arguments
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    # args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load('simple_urban_comm.py').Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.info, shared_viewer = True, discrete_action_space=True)
    # render call to create viewer window (necessary only for interactive policies)
    # env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    q_table = {}
    # policies = [GreedyPolicy(env, i, q_table) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    # print(f'call scenario.observation: {obs_n}')
    for j in range(100):
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_i = policy.action(obs_n[i], j)
            print(f"policy.action(obs_n[{i}]): {act_i}")
            act_n.append(act_i)
            print("-------分割线-------\n")

        # step environment
        # obs_n, reward_n, done_n, info_n = env.step(act_n)
        # print(f'info_n: {info_n}')
        print(f'act_n: {act_n}')
        print("\n分割线-------\n")
        # break
        # render all agent views
        # env.render()
        # time.sleep(0.5)
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
