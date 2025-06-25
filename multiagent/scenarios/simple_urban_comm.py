import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import utils.config as conf
import utils.util as utils
from utils.channel_model import Channel


class Scenario(BaseScenario):
    def __init__(self, seed=None):
        # 如果seed不为空时，将np.random.seed设为seed
        # 　这样在测试时，可以产生重复的数据
        if seed is not None:
            np.random.seed(seed)

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_p = conf.DIM_P
        world.dim_c = conf.DIM_C

        num_landmarks = conf.NUM_LANDMARKS
        script_agents = num_landmarks * conf.NUM_TERMINALS
        policy_agents = num_landmarks
        num_agents = script_agents + policy_agents

        # 是否所有的agent采用同一个reward进行训练
        world.collaborative = False

        # add agents
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            if i < policy_agents:
                agent.id = i
                agent.name = 'agent-%d' % i
                agent.movable = True
                agent.silent = False
                agent.max_speed = conf.MXA_SPEED
            else:
                agent.id = i - policy_agents
                agent.name = f'terminal-{i - policy_agents}'
                agent.movable = False  # 设置为不可移动，防止随机行走模型与环境施加的力相互影响。
                agent.silent = True
            agent.collide = False
            agent.size = conf.AGENT_SIZE if i < policy_agents else conf.TERMINAL_SIZE
            # agent.policy = True if i < policy_agents else False
            # random_walk 赋给action_callback后会直接调用
            agent.action_callback = None if i < policy_agents else utils.random_walk(world, agent)
            agent.draw_line = True if i < policy_agents else False
            # 通信范围
            agent.radius = conf.AGENT_RADIUS if i < policy_agents else conf.TERMINAL_RADIUS

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'landmark-%d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = conf.LANDMARK_SIZE
            # 地标范围
            landmark.radius = conf.LANDMARK_RADIUS
            # 保存连接关系
            landmark.contain = []

        # make initial conditions
        self.reset_world(world)
        return world

    # 在每一个step更新world的参数
    def reset_world(self, world, reset=True):
        if reset:
            # random properties for agents
            for i, agent in enumerate(world.agents):
                agent.color = [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]][i]if i < len(world.policy_agents) else np.array([1, 0, 0])
            # random properties for landmarks
            for i, landmark in enumerate(world.landmarks):
                landmark.color = np.array([0.25, 0.25, 0.25])
            random_creat_agent(world)
            # 初始化信道连接模型
            world.channel = Channel(world.policy_agents, world.scripted_agents)

            # 保存初始连接关系
            utils.update_agent_link(world)
        return self.reset_world

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for land in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - land.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    # 碰撞检测
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # 在此设置奖励函数
    def reward(self, agent, world):
        # 每个agent有一个单独的reward: 连接到agent的所有terminal可实现的数据率总和
        reward = 0
        for j in agent.link:
            terminal_j = world.scripted_agents[j]
            reward += world.channel.achievable_data_rate(agent, terminal_j)
        return reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.scripted_agents:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.agents:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        # 记录对其他agent的观察
        comm = []
        other_pos = []
        for other in world.policy_agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # # todo 离散化
        # agent.state.p_pos = conf.DISCRETIEZER.discretize_point(agent.state.p_pos)
        ob = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + entity_pos + other_pos + comm)
        return ob

    def info(self, agent, world):
        res_map = {
            'id': agent.id,
            'name': agent.name,
            'link': agent.link,
            'state': {
                'p_pos': agent.state.p_pos,
                'p_vel': agent.state.p_pos,
                'c': agent.state.c
            },
            'size': agent.size,
            'max_speed': agent.max_speed,
            'silent': agent.silent,
            'action': {
                'u': agent.action.u,
                'c': agent.action.c
            },
            'belong': agent.belong,
        }
        return res_map


# 按规则随机生成坐标
def random_creat_agent(world):
    # 步骤1：生成landmarks个随机圆心坐标,这里为了方便，固定landmark的坐标
    # 固定坐标后，则不需要随机种子
    centers = conf.CENTERS[:, :world.dim_p]  # 做一下维度兼容
    # 设置随机种子以获得可重复的结果
    # np.random.seed(conf.RANDOM_SEED)

    # 步骤2：调整圆心位置以确保圆相切, 固定后已经相切
    # 以通信半径为圆心
    r = world.landmarks[0].radius - 0.1  # 0.1是边界控制条件,尽量不生成在边界外

    for i, landmark in enumerate(world.landmarks):
        landmark.state.p_pos = centers[i]
        landmark.state.p_vel = np.zeros(world.dim_p)

    # 步骤3：在每个圆内生成随机坐标
    num_points_per_circle = 10
    points = []
    # agent也需要在小区范围内随机生成
    agent_points = []
    for center in centers:
        # 生成landmark圆内的点，对于坐标超过1的，重新生成

        # 这里的+1是因为直接一次性生成num_points_per_circle + 1个点，将第一个点作为agent，其余为terminals
        circle_points = np.zeros((num_points_per_circle + 1, world.dim_p))
        for i in range(num_points_per_circle + 1):
            while True:
                point = np.random.uniform(-r, r, world.dim_p) + center
                # 生成时允许超出landmark的范围
                if not conf.IN_LANDMARKS_WHILE_CREATE:
                    circle_points[i] = point  # 直接接受该坐标,并退出
                    break  # 直接退出,不做以下范围判断
                if np.all(np.abs(point) <= 1):  # 检查所有坐标的绝对值是否小于等于1,超出地图边界的重新生成
                    distances = np.linalg.norm(center - point)
                    if distances <= r:  # 如果点到圆心的距离小于radius
                        circle_points[i] = point  # 接受该坐标,并退出
                        break
        # 单独处理三维的情况
        if world.dim_p > 2:
            # 将高度取绝对值
            circle_points[0, 2] = np.abs(circle_points[0, 2])
            # 高度进行限制
            circle_points[0, 2] = utils.border_control(circle_points[0], control_height=conf.HEIGHT_CONTROL)[2]
            # 将除了第一个点以外的第三列值设为0
            circle_points[1:, 2] = 0
        # 把生成的第一个坐标赋给agent
        agent_points.append(circle_points[0])
        # 将其余坐标添加到终端坐标数组中
        points.append(circle_points[1:, :])

    # 将所有点合并为一个数组
    all_points = np.concatenate(points, axis=0)

    # 将agent_points的坐标赋值给agent
    for i, agent in enumerate(world.policy_agents):
        agent.state.p_pos = agent_points[i]
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
        # 将地面终端设置连接至landmark
        agent.belong = i
        world.landmarks[i].contain.append(agent.id)

    # 给地面终端赋坐标
    for j, agent in enumerate(world.scripted_agents):
        agent.state.p_pos = all_points[j]
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

        # 将地面终端设置连接至landmark
        num = j // 10
        agent.belong = num
        # world.policy_agents[num].contain.append(agent.id)
        world.landmarks[num].contain.append(agent.id)
