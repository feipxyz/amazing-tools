
"""
天池陶瓷瑕疵的可视化
"""

import cv2
from matplotlib.pyplot import connect
import numpy as np
import argparse
import json
import os
from vis import (
    vis_bboxes
)

def vis_tianchi_ceram(json_file, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    fr = open(json_file, 'r')
    content = json.load(fr)
    fr.close()

    # {'name': [[x,y,w,h,c]] }

    info = dict()
    for item in content:
        box = item['bbox']
        name = item['name']
        category = item['category']

        bboxes = info.get(name, [])
        bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        bboxes.append([*bbox, category])
        info[name] = bboxes

    for i, (name, bboxes) in enumerate(info.items()):
        image_path = os.path.join(src_dir, name)
        image = cv2.imread(image_path)
        bboxes = [[elem[0], elem[1], elem[2], elem[3]] for elem in bboxes]

        image = vis_bboxes(image, bboxes)
        dst_path = os.path.join(dst_dir, name)
        cv2.imwrite(dst_path, image)

        if i % 50 == 0:
            print("{} has done".format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("demo")
    parser.add_argument('-l', '--label_file', type=str, default=None, help='label file')
    parser.add_argument('-s', '--src_dir', type=str, default=None, help='image_dir')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='output dir')
    args = parser.parse_args()

    vis_tianchi_ceram(args.label_file, args.src_dir, args.dst_dir)





