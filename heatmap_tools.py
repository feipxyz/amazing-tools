import numpy as np
from itertools import product as product

class HeatMapTool(object):
    def __init__(self):
        pass

    def gaussian_2d(self, shape, sigma=1):
        """
        from centernet
        得到一个二维高斯核
        @param shape:
        @param sigma:
        @return:
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self, det_size, min_overlap=0.7):
        """
        from centernet
        高斯核半径
        @param det_size:
        @param min_overlap:
        @return:
        """
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        """
        from centernet
        得到heatmap，如果连个heatmap区域重叠，直接取最大
        @param heatmap:
        @param center:
        @param radius:
        @param k:
        @return:
        """
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        # left, right, top, bottom表示中心点向左、向右、向上、向下的距离
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            # 两个heatmap区域重叠，直接取最大
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


if __name__ == "__main__":
    pass
    json_file = "yuenan.json"
    root = "/Users/feipeng/Pictures/yuenan"
    json_file = "yinni.json"
    root = "/Users/feipeng/Pictures/yinni"
    # visual(json_file, root)