
"""
labelme标注文件可视化
"""

import cv2
import numpy as np
import argparse
import json
import os

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ONNX Location Demo")
    parser.add_argument('-l', '--label_list', type=str, default=None, help='label list')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='output dir')
    parser.add_argument('-s', '--src_dir', type=str, default=None, help='image_dir')
    args = parser.parse_args()

    vis_polygon_batch(args.label_list, args.src_dir, args.dst_dir)





