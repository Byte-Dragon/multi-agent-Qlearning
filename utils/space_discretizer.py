class SpaceDiscretizer:
    """
    将三维空间离散化为等体积子空间的处理器（原点在左上角），支持科学计算精度控制

    参数:
    w (int): X轴划分份数(≥1)
    l (int): Y轴划分份数(≥1)
    h (int): Z轴划分份数(≥1)
    precision (int): 输出坐标的小数位数(≥0)
    """

    def __init__(self, w: int, l: int, h: int, precision: int = 6, flag: bool = False):
        # 验证输入有效性
        if not all(isinstance(n, int) for n in (w, l, h)):
            raise TypeError("划分份数必须是整数")
        if not all(n >= 1 for n in (w, l, h)):
            raise ValueError("划分份数必须大于0")
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("精度必须是大于等于0的整数")

        self.w = w
        self.l = l
        self.h = h
        self.precision = precision

        # 计算每个维度的步长(子空间边长)
        self.dx = 1.0 / w
        self.dy = 1.0 / l
        self.dz = 1.0 / h
        self.flag = flag

    def discretize_point(self, point: tuple) -> tuple:
        """
        计算坐标点对应的子空间中心点（原点在左上角）

        参数:
        point (tuple): 原始坐标(x, y, z)，范围[0,1]
        flag (bool): 离散化开关
            True - 返回离散化坐标
            False - 返回原始坐标

        返回:
        tuple: 目标坐标(x, y, z)，精度根据precision参数控制
        """
        x, y, z = point

        # 直接返回原始坐标（带精度控制）
        if not self.flag:
            return self._apply_precision((x, y, z))

        # 坐标边界约束（确保在[0,1]范围内）
        x = max(0.0, min(x, 1.0))
        y = max(0.0, min(y, 1.0))
        z = max(0.0, min(z, 1.0))

        # 计算子空间索引（注意y轴方向反转）
        idx_x = min(int(x / self.dx), self.w - 1)
        idx_y = min(int((1.0 - y) / self.dy), self.l - 1)  # y轴反转处理
        idx_z = min(int(z / self.dz), self.h - 1)

        # 计算子空间中心点坐标（注意y轴方向反转）
        center_x = (idx_x + 0.5) * self.dx
        center_y = 1.0 - (idx_y + 0.5) * self.dy  # y轴反转处理
        center_z = (idx_z + 0.5) * self.dz

        return self._apply_precision((center_x, center_y, center_z))

    def _apply_precision(self, point: tuple):
        """
        应用精度控制到坐标点

        参数:
        point (tuple): 原始坐标(x, y, z)

        返回:
        tuple: 精度处理后的坐标(x, y, z)
        """
        if self.precision == 0:
            # 返回整数坐标
            return tuple(round(coord) for coord in point)

        # 返回指定精度的浮点数
        return list(round(coord, self.precision) for coord in point)


# 使用示例
if __name__ == "__main__":
    # 创建离散化处理器：空间范围1x1x1，划分3x2x4份，精度8位小数
    discretizer = SpaceDiscretizer(3, 2, 4, precision=8)

    # 测试点1：正常点
    test_point = (0.25, 0.75, 0.6)
    print(f"原始坐标: {test_point}")
    print(f"离散化坐标: {discretizer.discretize_point(test_point)}")

    # 测试点2：左上角点
    top_left = (0.0, 0.0, 0.0)
    print(f"\n左上角点: {top_left}")
    print(f"离散化坐标: {discretizer.discretize_point(top_left)}")

    # 测试点3：右下角点
    bottom_right = (1.0, 1.0, 1.0)
    print(f"\n右下角点: {bottom_right}")
    print(f"离散化坐标: {discretizer.discretize_point(bottom_right)}")

    # 测试精度控制
    print("\n测试不同精度:")
    for prec in [0, 2, 4, 8]:
        discretizer = SpaceDiscretizer(3, 2, 4, precision=prec)
        result = discretizer.discretize_point((0.123456789, 0.987654321, 0.555555555))
        print(f"精度 {prec}: {result}")

    # 测试flag关闭
    print(f"\n关闭离散化: {discretizer.discretize_point(test_point, False)}")