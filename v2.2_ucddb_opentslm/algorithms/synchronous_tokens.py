"""M01 根据模块顺序展开通道与时间轴。"""


def flatten_tokens(projected, synchronous):
    if synchronous:
        projected = projected.transpose(1, 2)
    return projected.flatten(1, 2)
