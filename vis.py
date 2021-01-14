import cv2


def vis_bboxes(image, bboxes):
    drawed = image.copy()
    for bbox in bboxes:
        bbox = [int(elem) for elem in bbox]
        x, y, w, h = bbox
        drawed = cv2.rectangle(drawed, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return drawed 


