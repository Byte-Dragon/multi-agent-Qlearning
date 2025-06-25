# 按规则随机生成坐标
'''
def random_creat_agent(self, world):
    # 设置随机种子以获得可重复的结果
    np.random.seed(conf.RANDOM_SEED)
    num_center = len(world.landmarks)
    # 步骤1：生成landmarks个随机圆心坐标
    centers = np.random.rand(num_center, world.dim_p)
    # centers = conf.CENTERS
    # 步骤2：调整圆心位置以确保圆相切
    # 以通信半径为圆心
    r = world.landmarks[0].radius
    # 计算每对圆心之间的距离
    # distances = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
    # 调整圆心位置以确保它们之间的距离大约为 2r
    # for i in range(num_center):
    #     for j in range(i + 1, num_center):
    #         diff = centers[j] - centers[i]
    #         dist = np.linalg.norm(diff)
    #         if dist < 2 * r:
    #             # 将一个圆心向外移动，直到距离为 2r
    #             centers[j] += (2 * r - dist) * diff / dist
    for i, landmark in enumerate(world.landmarks):
        landmark.state.p_pos = centers[i]
        landmark.state.p_vel = np.zeros(world.dim_p)

    # 步骤3：在每个圆内生成随机坐标
    num_points_per_circle = 10
    points = []
    # agent也需要在小区范围内随机生成
    agent_points = []
    for center in centers:
        # 随机生成的坐标有问题
        # circle_points = np.random.uniform(-r, r, (num_points_per_circle+1, world.dim_p)) + center

        # 生成圆上的点，对于坐标超过1的，重新生成
        circle_points = np.zeros((num_points_per_circle + 1, world.dim_p))
        for i in range(num_points_per_circle + 1):
            while True:
                point = np.random.uniform(-r, r, world.dim_p) + center
                if np.all(np.abs(point) <= r):  # 检查所有坐标的绝对值是否小于等于r
                    circle_points[i] = point
                    break
        # 单独处理三维的情况
        if world.dim_p > 2:
            # 将高度取绝对值
            circle_points[0, 2] = np.abs(circle_points[0, 2])
            # 将除了第一个点以外的第三列值设为0
            circle_points[1:, 2] = 0
        # 把生成的第一个坐标赋给agent
        agent_points.append(circle_points[0])
        points.append(circle_points[1:, :])

    # 将所有点合并为一个数组
    all_points = np.concatenate(points, axis=0)
    # 打印生成的点
    print(f'Generated points:\n{all_points},长度{len(all_points)}')

    # 将agent_points的坐标赋值给agent
    for i, agent in enumerate(world.policy_agents):
        agent.state.p_pos = agent_points[i]
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
    # 将points的坐标赋值给地面终端
    # print(f"ccc{len(world.scripted_agents)}")
    for j, agent in enumerate(world.scripted_agents):
        # try:
        #     agent.state.p_pos = points[j]
        # except Exception:
        #     print(f"{j}")
        agent.state.p_pos = all_points[j]
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
'''