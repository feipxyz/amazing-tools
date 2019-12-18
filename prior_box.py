import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    """
    RetinaFace中的预定义框生成
    {
        'min_sizes': [[10, 20], [32, 64], [128, 256]],
        'steps': [8, 16, 32],
        'clip': False
    }
    """
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        """
        anchor是在相对于原图的预定义框，注意坐标都缩放成了0-1之间
        :return:
        """
        anchors = []
        for k, f in enumerate(self.feature_maps):   # 遍历所有的feature map
            min_sizes = self.min_sizes[k]           # 对应feature map的预定义框大小
            for i, j in product(range(f[0]), range(f[1])):  # 针对每个特征图遍历坐标
                for min_size in min_sizes:                  # 预定义框的大小
                    # 预定义框大小放缩的0-1之间
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    # 中心点的坐标，放缩在0-1之间
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
