# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import math
import sys
import torch
from math import fabs
from math import radians
from math import sin
from math import cos


class ImageTools(object):
    """
    centernet中的图像变换
    """
    def __init__(self):
        pass

    def _blend(self, alpha, image1, image2):
        """
        from centernet
        @param alpha:
        @param image1:
        @param image2:
        @return:
        """
        image1 *= alpha
        image2 *= (1 - alpha)
        image1 += image2
        return image1

    def flip(self, image):
        """
        from centernet
        翻转图像
        :param image:
        :return:
        """
        return image[:, :, ::-1].copy()

    def shift(self, image, x, y):
        """
        from centernet
        图形平移
        :param image: 
        :param x: 
        :param y: 
        :return: 
        """
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return shifted

    def grayscale(self, image):
        """
        from centernet
        得到灰度图像
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def saturation(self, data_rng, image, gs, gs_mean, var, inplace=True):
        """
        from centernet
        饱和度变化
        :param data_rng: 随机数种子
        :param image: 原图 np.float32 [h, w c]
        :param gs: 灰度图 np.float32 [h, w]
        :param gs_mean: 灰度均值 const
        :param var: 变化范围 [0.0-1.0]
        """
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        if inplace:
            return self._blend(alpha, image, gs[:, :, None])
        else:
            new_image = image.copy()
            return self._blend(alpha, new_image, gs[:, :, None])

    def brightness(self, data_rng, image, gs, gs_mean, var=0.4, inplace=True):
        """
        from centernet
        亮度变化
        :param data_rng: 随机数种子
        :param image: 原图 np.float32 [h, w c]
        :param gs: 灰度图 np.float32 [h, w]
        :param gs_mean: 灰度均值 const
        :param var: 变化范围 [0.0-1.0]
        """
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        if inplace:
            image *= alpha
            return image
        else:
            new_image = image.copy()
            new_image *= alpha
            return new_image

    def contrast(self, data_rng, image, gs, gs_mean, var, inplace=True):
        """
        from centernet
        对比度变化
        :param data_rng: 随机数种子
        :param image: 原图 np.float32 [h, w c]
        :param gs: 灰度图 np.float32 [h, w]
        :param gs_mean: 灰度均值 const
        :param var: 变化范围 [0.0-1.0]
        """

        alpha = 1. + data_rng.uniform(low=-var, high=var)

        if inplace:
            return self._blend(alpha, image, gs_mean)
        else:
            new_image = image.copy()
            return self._blend(alpha, new_image, gs_mean)

    def color_aug(self, data_rng, image, inplace=True):
        """
        from centernet
        图像色彩变化，包含亮度、对比度、饱和度
        :param data_rng: 随机数种子
        :param image: 原图 np.float32 [h, w c]
        :param gs: 灰度图 np.float32 [h, w]
        :param gs_mean: 灰度均值 const
        :param var: 变化范围 [0.0-1.0]
        """

        functions = [self.brightness, self.contrast, self.saturation]
        random.shuffle(functions)

        gs = self.grayscale(image)
        gs_mean = gs.mean()
        new_image = image
        if not inplace:
            new_image = image.copy()
        for f in functions:
            new_image = f(data_rng, new_image, gs, gs_mean, 0.4)
        return new_image

    def _pad_to_ratio(self, image, rgb_mean, ratio):
        """
        pad到指定的宽高比
        :param image:
        :param rgb_mean:
        :param ratio:
        :return:
        """
        height, width, _ = image.shape
        if height/width == ratio:
            return image

        new_height = height
        new_width = width
        pad_height = 0
        pad_width = 0
        if height/width > ratio:
            new_width = math.ceil(new_height / ratio)
            pad_width = (new_width - width) // 2
        else:
            new_height = math.ceil(width * ratio)
            pad_height = (new_height - height) // 2

        image_t = np.empty((new_height, new_width, 3), dtype=image.dtype)
        image_t[:, :] = rgb_mean
        image_t[pad_height : pad_height + height, pad_width : pad_width + width] = image
        return image_t

    def _get_dir(self, src_point, rot_rad):
        """
        from centernet
        @param src_point:
        @param rot_rad:
        @return:
        """
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def _get_3rd_point(self, a, b):
        """
        from centernet
        @param a:
        @param b:
        @return:
        """
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_affine_transform(self, center, scale, rot, output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
        """
        from centernet
        centernet中使用的仿射变换，这个其实可以起到pad的作用
        使用示例:
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0
        input_res = 512
        trans_input = img_tool.get_affine_transform(
            c, s, rot, [input_res, input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (input_res, input_res),
                             flags=cv2.INTER_LINEAR)
        @param center:
        @param scale:
        @param rot:
        @param output_size:
        @param shift:
        @param inv:
        @return:
        """
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self._get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def rotate_keep(self, degree, width, height):
        """
        无损图像的旋转
        @param degree:
        @param width:
        @param height:
        @return: 旋转矩阵，新的宽高
        """
        height_new = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        width_new = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        mat_rotation[0, 2] += (width_new - width) / 2  # ?????重点在这步，目前不懂为什么加这步
        mat_rotation[1, 2] += (height_new - height) / 2  # ?????重点在这步
        return mat_rotation, width_new, height_new


def concat_images_3channle(images):
    max_height = 0 
    max_width = 0
    for image in images:
        height, width, _ = image.shape
        max_height = max(height, max_height)
        max_width = max_width + width

    res = np.zeros((max_height, max_width, 3), np.uint8)
    cur_len = 0
    for image in images:
        height, width, _ = image.shape
        res[0:height, cur_len:cur_len+width, :] = image
        cur_len = cur_len + width

    return res 


if __name__ == "__main__":
    img_tool = ImageTools()
    # image_path = "/Users/feipeng/Pictures/cat.jpg"
    # image_path = "/Users/feipeng/Pictures/weixin/mmexport1552661278016.jpg"
    image_path = "/Users/feipeng/Pictures/weixin/mmexport1552055359285.jpg"
    img = cv2.imread(image_path)
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0
    input_res = 512
    trans_input = img_tool.get_affine_transform(
        c, s, rot, [input_res, input_res])
    inp = cv2.warpAffine(img, trans_input,
                         (input_res, input_res),
                         flags=cv2.INTER_LINEAR)
    cv2.imshow("result", inp)
    cv2.waitKey()
    # data_rng = np.random.RandomState(123)
    # data_rng = np.random.RandomState()
    # inp = (image.astype(np.float32) / 255.)
    # output = img_tool.color_aug(data_rng, inp, inplace=True)

    # cv2.imshow("result", inp)
    # cv2.waitKey()
    # cv2.imshow("result", image)
    # cv2.waitKey()

    # image_bbox_tool.random_crop(image, bbox, labels, landmarks, 640)
    # cv2.imshow("result", new_img)
    # cv2.waitKey()
    # roi = np.array([[1, 1, 19, 19], [6, 6, 10, 10]])
    # image_bbox_tool.matrix_iof(bbox, roi)
    # b = image_bbox_tool.matrix_iou(bbox, roi)
    # print(b.shape)





