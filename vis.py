import cv2


def vis_bboxes(image, bboxes):
    for bbox in bboxes:
        bbox = [int(elem) for elem in bbox]
        x, y, w, h = bbox
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


