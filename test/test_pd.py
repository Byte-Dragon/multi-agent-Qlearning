import pandas as pd
import numpy as np

# 假设观察结果的可能值是一组numpy数组对象
# 这里我们随机生成一些示例数据作为观察结果
observations = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

# 动作的数量
num_actions = 4  # 假设有4个可能的动作

# 创建Q表
q_table = pd.DataFrame(0, index=observations, columns=range(num_actions))

# print(q_table)


# # 创建一个DataFrame
# df = pd.DataFrame({
#     '0': [1, 2, 3],
#     '1': [4, 5, 6]
# }, index=[np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])])
# # print([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])])
# print(f'初始： {df}\n')
#
# # 使用loc通过name获取Series
# series_a = df.loc[tuple([1,4,7]), :]
# print(f'后来： {series_a[1]}\n')

# # 使用xs通过name获取Series
# series_a_xs = df.xs('A', axis=1, level=0)
# print(series_a_xs)
key = tuple([1.00, 2.0323, 4])
key2 = tuple([1.00, 2.0323, 4, 3])
q = [{key: np.zeros(7), key2: np.zeros(8)}, {'1': [1,2,3], '2': [2,3,4]}]
print(q)
print(q[0][key][5])
print(q[0][key2])
print(q[1])
print(q[1]['1'])
print(q[1]['1'][0])
