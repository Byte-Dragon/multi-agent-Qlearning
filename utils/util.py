# 随机行走模型
import math
import random
import numpy as np
import utils.config as conf


# 返回一个以gp
def get_grid_center(position, grid_size):
    """
    返回给定坐标position所处的网格中心坐标。
    如果position位于边界上，返回左上角网格的中心坐标。
    """
    x, y = position
    # 向下取整到最近的网格中心
    grid_x_index = int(x // grid_size)
    grid_y_index = int(y // grid_size)

    # 计算网格中心坐标
    grid_center_x = grid_x_index * grid_size + grid_size / 2
    grid_center_y = grid_y_index * grid_size + grid_size / 2

    return (grid_center_x, grid_center_y)


# 地面终端的随机行走模型
def random_walk(world, agent, step=False):
    # step防止在reset_world时，将random_walk传递给action_callback时调用

    if step:
        # 遍历world中的所有agent
        v_max = conf.TERMINAL_V_MAX
        # for agent in world.scripted_agents:
        # 随机选择一个角度a，范围从0到2π
        angle = random.uniform(0, 2 * math.pi)
        # 随机选择一个不大于v_max的速度
        speed = random.uniform(0, v_max)

        # 计算新的位置
        new_x = agent.state.p_pos[0] + speed * math.cos(angle)
        new_y = agent.state.p_pos[1] + speed * math.sin(angle)
        # clip_x = 0.0
        # clip_y = 0.0
        # 限定坐标值在landmark的radius范围内
        # 如果超出，则将其限制在边界上
        # 有个问题，如何判断该terminal所在的边界是哪里？
        # 或者说，判断该以谁为边界
        landmark = world.landmarks[agent.belong]
        if world.dim_p > 2:
            pos = [new_x, new_y, 0]
        else:
            pos = [new_x, new_y]

        # 进行边界控制
        r = landmark.radius - conf.CONTROL_PARA
        origin = landmark.state.p_pos
        # 更新agent的坐标
        agent.state.p_pos = boundary_control(pos, origin, r)
        # 更新连接状态
    return random_walk


# 对以origin为圆心,radius为半径的区域内一点position进行边界控制
def boundary_control(position, origin, radius):
    # 不能超过地图的边界
    position = border_control(position)

    # 是否进行范围控制
    if conf.ALLOW_OUT:
        return position
    else:
        r = radius
        result = np.zeros(len(position))
        result[:2] = position[:2]
        distances = np.linalg.norm(np.array(origin) - np.array([position]))
        if distances > r:
            # 计算点 (new_x, new_y) 到圆心 origin 的单位向量
            unit_vector = (np.array(position) - np.array(origin)) / distances
            # 将点 (new_x, new_y) 限制在圆的边界上
            result[0] = origin[0] + r * unit_vector[0]
            result[1] = origin[1] + r * unit_vector[1]
        return result


# 地图边界控制,坐标绝对值不能超过1
def border_control(position, control_height=False):
    for i in range(len(position)):
        if abs(position[i]) > 1:
            border = 1 - conf.CONTROL_PARA
            position[i] = border if position[i] > 0 else -border

    # 是否进行高度的控制
    if control_height:
        position = height_control(position)

    return position


# 对高度进行限制，限制在[lowest, highest]区间内
def height_control(position, lowest=conf.LOWEST, highest=conf.HIGHEST):
    if len(position) > 2:
        position[2] = max(min(position[2], highest), lowest)
    return position


# 更新world当前的连接关系
def update_agent_link(world):

    # 保存初始连接关系
    res_map, _ = world.channel.link_matrix()
    for agent in world.policy_agents:
        agent.link = res_map[agent.name]
