import os
import cv2
import numpy as np
from pathlib import Path

import time

from sort import Sort
from utils import init_onnx, process_image, run, preprocess, INP_SIZE, postprocess


def timing_val(func):
    def wrapper(*arg, **kw):
        """source: http://www.daniweb.com/code/snippet368.html"""
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__
    return wrapper


@timing_val
def run_model(model, inp, n=1):
    for _ in range(n):
        run(model, inp)


@timing_val
def run_preprocess(image):
    preprocess(image)


@timing_val
def run_postprocess(out, scale_x, scale_y):
    postprocess(out, scale_x, scale_y)

@timing_val
def run_detection(model, img, n=1):
    boxes = None
    for _ in range(n):
        orig_h, orig_w, _ = img.shape
        scale_x = orig_w / INP_SIZE[0]
        scale_y = orig_h / INP_SIZE[1]
        inp = preprocess(img)
        out = run(model, inp)
        boxes = postprocess(out, scale_x, scale_y)
    return boxes


@timing_val
def run_tracker(model, sort, images, n=30):
    for img in images[:n]:
        orig_h, orig_w, _ = img.shape
        scale_x = orig_w / INP_SIZE[0]
        scale_y = orig_h / INP_SIZE[1]
        inp = preprocess(img)
        out = run(model, inp)
        boxes = postprocess(out, scale_x, scale_y)
        trackers = np.copy(sort.update(boxes))


def main():
    yolo = init_onnx("./weights/mot17-01-frcnn.onnx", gpu=False)
    mot_tracker = Sort(max_age=15, min_hits=1, iou_thresh=0.3)  # create instance of the SORT tracker

    images_dir = Path('./speed_test/images/')
    images_paths = sorted(next(os.walk(images_dir))[2])
    img_path = './speed_test/images/000001.jpg'
    img = cv2.imread(str(img_path))
    # images = [cv2.imread(str(images_dir / pth)) for pth in images_paths]

    orig_h, orig_w, _ = img.shape
    scale_x = orig_w / INP_SIZE[0]
    scale_y = orig_h / INP_SIZE[1]

    # res = run_preprocess(img)
    inp = preprocess(img)
    # out = run(yolo, inp)
    # res = run_postprocess(out, scale_x, scale_y)

    res = run_model(yolo, inp, n=1)
    # res = run_detection(yolo, img, n=1)
    # res = run_tracker(yolo, mot_tracker, images, n=15)


    print('%s took %0.3fms.' % (res[2], res[0]*1000))


if __name__ == '__main__':
    main()
