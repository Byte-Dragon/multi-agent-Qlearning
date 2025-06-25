import math
import numpy as np

# agent_pos = [1,1,0]
# terminal_pos = [1,1,0]
# print(np.array(agent_pos[:-1]))
# print(np.array(agent_pos[:-1]) - np.array(terminal_pos[:-1]))
# print(np.linalg.norm(np.array(agent_pos[:-1]) - np.array(terminal_pos[:-1])) )

# # 假设有一个三维数组
# arr = np.array([
#     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 第一深度层
#     [[10, 11, 12], [13, 14, 15], [16, 17, 18]],  # 第二深度层
#     [[19, 20, 21], [22, 23, 24], [25, 26, 27]]  # 第三深度层
# ])
#
# # 正确的获取第一行的所有列的第三维数据
# first_row_third_dim = arr[0, :, 2]  # 0代表第一层，:代表所有行，2代表第三列
# print(first_row_third_dim)  # 输出: [ 3  6  9]
#
# test = arr[0]
# print(test)
#
# test2 = test[:, 2]
# print(test2)
#
# test3 = test[:][2]
# print(test3)
tmp = [-1, -1, -1]
# for i in range(3):
#     tmp.append(-1)
t = np.all(np.array(tmp) == -1)
print(t)
