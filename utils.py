import cv2
import numpy as np
import torch  # for cudnn
import onnxruntime as ort
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from config import NMS_THRESH

INP_SIZE = (640, 640)


def iou(box1, box2):
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    overlap_area = x_overlap * y_overlap
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    total_area = area1 + area2 - overlap_area
    return overlap_area / total_area


def nms(boxes, thr=0.5):
    # sort by confidence
    if boxes.shape[0] == 0:
        return np.empty((0, 5))
    boxes = list(boxes[boxes[:, -1].argsort()])
    final_boxes = []
    while len(boxes) > 0:
        final_boxes.append(boxes.pop(0))
        boxes = [box for box in boxes if iou(box, final_boxes[-1]) <= thr]
    return np.array(final_boxes)


def init_onnx(model_path):
    ep_list = ['CUDAExecutionProvider']
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=ep_list)
    return sess


def preprocess(img, bgr2rgb=True):
    img_copy = np.copy(img)
    if bgr2rgb:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_copy = cv2.resize(img_copy, INP_SIZE)
    img_copy = np.transpose(img_copy, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_copy = np.expand_dims(img_copy, axis=0)
    img_copy = np.ascontiguousarray(img_copy / 255)
    return img_copy


def run(model, inp):
    outputs = model.run(None, {"images": inp})
    return outputs[0][0]


def postprocess(raw_out, scale_x: float = 1.0, scale_y: float = 1.0):
    correct = []
    for pred in raw_out.T:
        x, y, w, h, score = pred
        if score > 0.3:
            x1 = int(x - w / 2) * scale_x
            x2 = int(x + w / 2) * scale_x
            y1 = int(y - h / 2) * scale_y
            y2 = int(y + h / 2) * scale_y
            correct.append([x1, y1, x2, y2, score])
    correct = np.array(correct)
    final_boxes = nms(correct, NMS_THRESH)
    return final_boxes


def process_image(model, img_path: Path):
    img = cv2.imread(str(img_path))
    orig_h, orig_w, _ = img.shape
    scale_x = orig_w/INP_SIZE[0]
    scale_y = orig_h/INP_SIZE[1]
    inp = preprocess(img)
    out = run(model, inp)
    boxes = postprocess(out, scale_x, scale_y)
    return boxes


def orient_area(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
