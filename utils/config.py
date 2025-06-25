import numpy as np

# 定义训练参数
from utils.space_discretizer import SpaceDiscretizer

LEARNING_RATE = 0.1  # Qleaning的学习率。alpha, if use value function approximation, we can ignore it
DISCOUNT_FACTOR = 0.90  # 折扣率 decay factor
EPISODES = 50  # 迭代次数，也就是开始10000次游戏
STEPS = 500  # 每次游戏进行的最大步数
EPSILON = 0.9  # e-greedy参数
RENDER = False
# RENDER = True
# 定义一些配置参数

# 随机数种子
RANDOM_SEED = 19

# 世界参数
DIM_P = 3
DIM_C = 1
NUM_LANDMARKS = 3
NUM_TERMINALS = 10
NUM_ACTIONS = DIM_P * 2 + 1

WORLD_SIZE = [800, 800]
DISCRETIEZER_FLAG = True
# DISCRETIEZER_FLAG = True
# 创建离散化处理器：空间范围1x1x1，划分3x2x4份，精度8位小数
DISCRETIEZER = SpaceDiscretizer(800, 800, 800, precision=8, flag=DISCRETIEZER_FLAG)

#  ###### 绘制参数 #######
# 可视化圆半径
AGENT_SIZE = 0.03
TERMINAL_SIZE = 0.015
LANDMARK_SIZE = 0.05

# 是否绘制小区圆和外圈圆
DRAW_LANDMARKS = False

# 通信半径
AGENT_RADIUS = 0.5
TERMINAL_RADIUS = 0.15
LANDMARK_RADIUS = 0.5
IF_DISTANCE_WHILE_CLOSET_AGENT = True

# 小区初始的坐标
CENTERS = np.array([[0, 0.87 - 0.5, 0],  # 0.87为根号3/2
           [-0.5, -0.5, 0],
           [0.5, -0.5, 0]])

# agent移动的尺度，物理力的大小
DELTA_FORCE = 1.5

TERMINAL_V_MAX = 0.05
AGENT_SENSITIVITY = 2.0
# agent 的action step size
MXA_SPEED = 0.2


# 是否允许terminal走出landmark的radius范围
IN_LANDMARKS_WHILE_CREATE = True
ALLOW_OUT = True
CONTROL_PARA = 0.01  # 边界控制参数,与边界的距离
HIGHEST = 0.2
LOWEST = 0.1
HEIGHT_CONTROL = True

# 信道模型相关的训练超参数
A = 0.76  # subUrban
B = 0.06  # subUrban
XI_LOS = 0.1  # dB
XI_NLOS = 21  # dB

# 无人机的默认传输强度，transmit power
TRANSMIT_POWER = 30  # dBm

# 无人机分配的带宽total bandwidth
TOTAL_BANDWIDTH = 200  # kHz

# 噪声强度，noise power
NOISE_POWER = -120  # dBm


# 动作空间的action size
DELTA_R = 0.01
DELTA_THETA = 0.01
DELTA_ALPHA = 0.01
