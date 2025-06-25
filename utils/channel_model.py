import numpy as np
import math
import utils.config as conf


# 定义信道模型
class Channel(object):
    # 初始化，传入参数：agents, terminals
    def __init__(self, agents, terminals):
        self.agents = agents
        self.terminals = terminals
        # _, self.matrix = self.link_matrix()

    # 在分布式环境下，返回有关world的所有信息，需要有一个控制终端，
    # 在此只是模拟一下建立通信的过程，将该现实过程进行了简化
    # 判断连接到那个一个agent
    def link_matrix(self, by_signal=True):
        # 结果词典
        result_map = {agent.name: [] for agent in self.agents}
        # 连接矩阵，存在连接关系则置为1
        result_matrix = np.zeros((len(self.agents), len(self.terminals), 2))
        for terminal in self.terminals:
            tmp, best = self.closest_agent_by_signal(terminal) if by_signal else self.closest_agent_by_distance(
                terminal)
            # 　如果最近的agent非空的话
            if tmp is not None:
                # result_matrix[tmp.id][terminal.id] = 1
                result_matrix[tmp.id][terminal.id][0] = 1
                result_matrix[tmp.id][terminal.id][1] = best
                result_map[tmp.name].append(terminal.id)
        return result_map, result_matrix

    # 返回距离最近的某一terminal最近的agent
    def closest_agent_by_distance(self, terminal):
        # 初始化最小距离和最近agent
        min_distance = float('inf')  # 正无穷大
        closest_agent = None

        # 获取terminal的坐标
        terminal_pos = np.array(terminal.state.p_pos)

        # 遍历所有agents
        for agent in self.agents:
            # 获取agent的坐标
            agent_pos = np.array(agent.state.p_pos)

            # 计算terminal到agent的距离
            distance = np.linalg.norm(terminal_pos - agent_pos)

            # 如果当前距离小于已知的最小距离，则更新最小距离和最近agent
            if distance < min_distance:
                min_distance = distance
                closest_agent = agent

        return closest_agent, min_distance

    # 根据接收的信号强度，received signal power，返回terminal所需要连接到的agent
    # 计算信号强度最大的影响其实也是距离，但有概率不完全按照连接到距离最近的，
    # 现实中即存在 近却遮挡 的情况下信号肯定没有 远而无遮挡 的信号好
    # 超过通信范围距离的返回空
    def closest_agent_by_signal(self, terminal):
        # 初始化
        max_rec_sign = 0.0
        closest_agent = None
        tmp = []
        for agent in self.agents:
            signal = received_signal_power(agent.state.p_pos, terminal.state.p_pos)
            tmp.append(signal)

        # 是否进行通信距离的控制
        # 不进行控制时，循环只执行一次，直接返回最大的信号值所对应的agent
        # 进行控制时，循环最多执行len(self.agents)次，当找到一个满足通信距离范围的最大信号值时，退出循环
        while True:
            max_index = tmp.index(max(tmp))  # 返回最大值索引

            if conf.IF_DISTANCE_WHILE_CLOSET_AGENT:
                distance = np.linalg.norm(np.asarray(self.agents[max_index].state.p_pos) - np.asarray(terminal.state.p_pos))
                if distance > conf.AGENT_RADIUS:
                    tmp[max_index] = -1
                    if not np.all(np.array(tmp) == -1):
                        continue
                    else:  # 如果都不在通信范围内
                        break

            closest_agent = self.agents[max_index]
            max_rec_sign = tmp[max_index]
            break
        return closest_agent, max_rec_sign

    # 返回可实现的数据率Upsilon_i_j，the achievable data rate
    def achievable_data_rate(self, agent_i, terminal_j):
        terminal_pos = terminal_j.state.p_pos

        # 先计算SINR of user j associated with UAV-BS
        sum = 0
        pr_i_j = 0
        for i, agent in enumerate(self.agents):
            if agent is not agent_i:
                # 计算sum(Pr_k_j), k=1 to N 且 k!=i,
                # print(f"计算sum:{agent.name}, {received_signal_power(agent.state.p_pos, terminal_pos)}")
                sum += received_signal_power(agent.state.p_pos, terminal_pos)
            else:
                # 计算Pr_i_j
                pr_i_j = received_signal_power(agent_i.state.p_pos, terminal_pos)

        # the SINR
        gamma_i_j = pr_i_j / (sum + conf.NOISE_POWER ** 2)

        # 计算可实现的数据率
        m_i = len(agent_i.link)  # 连接到agent i 的users数量

        upsilon_i_j = (conf.TOTAL_BANDWIDTH / m_i) * math.log2(1 + gamma_i_j)
        # print(f'可实现的数据率{agent_i.name} to {terminal_j.name}:\n pr_ij: {pr_i_j}, sum: {sum}, M_i: {m_i}, Upsilon_ij: {upsilon_i_j}, gamma: {gamma_i_j}')
        return upsilon_i_j


# 返回LoS情况发生的概率， LoS：Line of Sight
def los_probability(agent_pos, terminal_pos):
    # a,b为拟合系数，fitting coefficients
    a = conf.A
    b = conf.B
    distance_x_y = np.linalg.norm(np.array(agent_pos[:-1]) - np.array(terminal_pos[:-1]))

    # 这里计算math.atan(agent_pos[2] / distance_x_y) - 15) ** b会出现复数
    # 为了简便，取复数的模作为结果
    p_los_ij = a * abs((math.atan(agent_pos[2] / distance_x_y) - 15) ** b)
    # print(f'los probability: {a}, {b}, {distance_x_y}, {p_los_ij}, {abs(p_los_ij)}')
    return p_los_ij


# 返回接收到的信号强度，received signal power
def received_signal_power(agent_pos, terminal_pos):
    # 计算eta_i_j： agent i 与 地面的userj之间的平均路径损失，average path loss
    p_los_ij = los_probability(agent_pos, terminal_pos)
    r_i_j = np.linalg.norm(np.asarray(agent_pos) - np.asarray(terminal_pos))
    distance = 20 * math.log10(r_i_j)
    eta_los_i_j = distance + conf.XI_LOS
    eta_nlos_i_j = distance + conf.XI_NLOS
    eta_i_j = p_los_ij * eta_los_i_j + (1 - p_los_ij) * eta_nlos_i_j
    # 计算信号强度
    # 这里假设每个无人机的传输强度Pt_i都是一样的，为conf.TRANSMIT_POWER
    pr_i_j = conf.TRANSMIT_POWER - eta_i_j
    # print(f'received signal power: {p_los_ij}, {r_i_j}, {distance}, {eta_los_i_j}, {eta_nlos_i_j}, {eta_i_j}, {pr_i_j}')
    return pr_i_j
