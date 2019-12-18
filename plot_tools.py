import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform


# def cvt2HeatmapImg(img):
#     img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
#     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#     return img

def draw_heatmap(heat_map, image=None, alpha=0.7, cmap='viridis', show=False, save=None):
    """
    可视化heatmap
    @param heat_map: (h, w)
    @param image: (h, w, c)
    @param alpha:
    @param cmap:
    @param show:
    @param save:
    @return:
    """
    heat_map_resized = heat_map
    if image is not None:
        height = image.shape[0]
        width = image.shape[1]
        heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    if image is not None:
        plt.imshow(image)

    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)


