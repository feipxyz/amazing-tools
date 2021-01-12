
"""
对labelme的标注文件进行增强, 得到训练文件
"""

from imgaug.augmentables import base
from imgaug.augmenters.geometric import Rot90
import numpy as np
import cv2
import json
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug
import argparse
import os
import glob

sometimes = lambda p, aug: iaa.Sometimes(p, aug)

def get_points_from_label(content):
    points = []
    for elem in content['shapes']:
        points.append(elem["points"])
    return points

def vis_polygon(image, points):
    # points (n, m, 2)
    points = np.array(points)
    points = points[:, :, np.newaxis, :].astype(np.int)
    # cv2.polylines(image, points, True, (128, 255, 0), thickness=2)
    cv2.polylines(image, points, True, (0, 0, 255), thickness=2)
    return image

def vis_polygon_from_json(image_path, json_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    fr = open(json_path, 'r')
    content = json.load(fr)
    fr.close()
    points = get_points_from_label(content)
    image = vis_polygon(image, points)

    return image

def vis_polygon_batch(label_list, image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(label_list, 'r') as fr:
        for i, label_path in enumerate(fr):
            label_path = label_path.strip()
            name = os.path.split(label_path)[-1]
            shortname = os.path.splitext(name)[0]

            image_path = os.path.join(image_dir, shortname + ".jpg")
            image = vis_polygon_from_json(image_path, label_path)

            save_image_path = os.path.join(output_dir, shortname+".jpg")
            cv2.imwrite(save_image_path, image)

            if i % 10 == 0:
                print("{} has done".format(i))

def convert_to_segment(mask, points, scalar):
    """
    @param image
    @param points [[x1, y1], [x2, y2], [x3, y3], [x1, y1]]
    """
    points = np.array(points)
    # cv2.fillConvexPoly(mask, points, scalar)
    cv2.fillPoly(mask, [points], scalar)

    return mask

def resize_by_scale(image, max_len):
    height, width = image.shape[0:2]
    scale = min(max_len/height, max_len/width) 
    
    if height < width:
        new_width = max_len 
        new_height = int(round(height * scale))
    elif height > width:
        new_width = int(round(width * scale))
        new_height = max_len
    else:
        new_width = max_len
        new_height = max_len

    # image = cv2.imresize(image, (new_width, new_height))
    # return image, scale
    return new_width, new_height


def get_segments(label_list, image_dir, output_dir, max_len, pixel_constant=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_seq = iaa.Sequential(
        [
            sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0.05, 0.4))),
            sometimes(0.3, iaa.GammaContrast(gamma=(0.8, 1.2))),
            sometimes(0.2, iaa.AddToSaturation(value=(-10, 10))),
            sometimes(0.2, iaa.GaussianBlur(sigma=(0.05, 0.3))),
        ],
        random_order=True
    )

    seq_origin = iaa.Sequential([
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="right-bottom")
    ])

    seq_lr = iaa.Sequential([
        iaa.Fliplr(),
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="right-bottom")
    ])

    seq_ud = iaa.Sequential([
        iaa.Flipud(),
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="right-bottom")
    ])

    seq_rot90 = iaa.Sequential([
        # iaa.Rotate(rotate=90),
        iaa.Rot90(k=1, keep_size=False),
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="right-bottom")
    ])

    seq_rot180 = iaa.Sequential([
        # iaa.Rotate(rotate=180),
        iaa.Rot90(k=2, keep_size=False),
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="right-bottom")
    ])

    seq_rot270 = iaa.Sequential([
        iaa.Rot90(k=3, keep_size=False),
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="right-bottom")
    ])

    seq_normal = iaa.Sequential([
        base_seq,
        iaa.PadToFixedSize(width=max_len, height=max_len, position="normal")
    ])

    seq_resize = iaa.Sequential([
        base_seq,
        iaa.Resize((max_len, max_len))
        # iaa.PadToFixedSize(width=max_len, height=max_len, position="normal")
    ])

    with open(label_list, 'r') as fr:
        for i, label_path in enumerate(fr):
            label_path = label_path.strip()
            name = os.path.split(label_path)[-1]
            shortname = os.path.splitext(name)[0]

            # get points 
            content = json.load(open(label_path, 'r'))
            points = get_points_from_label(content)
            psoi = Polygon(points[0]) 

            # get image
            
            image_path = glob.glob(os.path.join(image_dir, shortname + ".*"))[0]
            image = cv2.imread(image_path)

            # resize
            new_width, new_height = resize_by_scale(image, max_len)
            seq = iaa.Sequential([
                iaa.Resize({"height":new_height, "width":new_width}),
            ])
            image, psoi = seq(image=image, polygons=psoi)

            #
            image_aug, psoi_aug = seq_origin(image=image, polygons=psoi)
            mask = np.zeros((max_len, max_len), dtype=np.uint8)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_origion.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_origion.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            #
            image_aug, psoi_aug = seq_lr(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_horizontal.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_horizontal.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            #
            image_aug, psoi_aug = seq_ud(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_vertical.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_vertical.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            # 
            image_aug, psoi_aug = seq_rot90(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_rot90.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_rot90.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            #
            image_aug, psoi_aug = seq_rot180(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_rot180.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_rot180.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            #
            image_aug, psoi_aug = seq_rot270(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_rot270.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_rot270.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            #
            image_aug, psoi_aug = seq_normal(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_normal.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_normal.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            #
            image_aug, psoi_aug = seq_resize(image=image, polygons=psoi)
            mask_new = convert_to_segment(mask.copy(), psoi_aug.coords.round().astype(np.int), pixel_constant)
            save_image_path = os.path.join(output_dir, shortname+"_resize.jpg")
            save_label_path = os.path.join(output_dir, shortname+"_resize.png")
            cv2.imwrite(save_image_path, image_aug)
            cv2.imwrite(save_label_path, mask_new)

            # image_polys_aug = psoi_aug.draw_on_image(image_aug)
            # cv2.imwrite("output.jpg", image_polys_aug)
            # exit()

            if i % 10 == 0:
                print("{} has done".format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ONNX Location Demo")
    parser.add_argument('-l', '--label_list', type=str, default=None, help='label list')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='output dir')
    parser.add_argument('-s', '--src_dir', type=str, default=None, help='image dir')
    parser.add_argument('--max_len', type=int, default=320, help='max len')
    args = parser.parse_args()

    get_segments(args.label_list, args.src_dir, args.dst_dir, args.max_len, pixel_constant=1)





