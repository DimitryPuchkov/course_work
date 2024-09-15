from itertools import product
from typing import Optional
import torch  # for cudnn
import onnxruntime as ort
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

np.random.seed(0)
colours = np.random.randint(low=0, high=255, size=(32, 3))


def iou(box1: np.array, box2: np.array) -> float:
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    overlap_area = x_overlap * y_overlap
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    total_area = area1 + area2 - overlap_area
    return overlap_area / total_area


def nms(boxes: np.array, thr: float = 0.65) -> np.array:
    # sort by confidence
    if boxes.shape[0] == 0:
        return np.empty((0, 5))
    boxes = list(boxes[boxes[:, -1].argsort()])
    final_boxes = []
    while len(boxes) > 0:
        final_boxes.append(boxes.pop(0))
        boxes = [box for box in boxes if iou(box, final_boxes[-1]) <= thr]
    return np.array(final_boxes)


def linear_assignment(cost_matrix: np.array) -> np.array:
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.array, bb_gt: np.array) -> np.array:
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
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh
    )
    return o


def convert_bbox_to_z(bbox: np.array) -> np.array:
    """
    [x1,y1,x2,y2] => [x,y,s,r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.array, score: Optional[float] = None, score_threshold: Optional[float] = None) -> np.array:
    """
    [x,y,s,r] => [x1,y1,x2,y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):

    count = 0

    def __init__(self, bbox: np.array) -> None:
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.array) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.array:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.array:
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: np.array,
    trackers: np.array,
    iou_threshold: float = 0.3
) -> tuple[np.array, np.array, np.array]:
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age: int = 1, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets: np.array = np.empty((0, 5))) -> np.array:
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(
            to_del
        ):  # reverse order because index are changing when deliting element from list
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def init_onnx(model_path: str) -> ort.InferenceSession:
    EP_list = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession("./runs/detect/train2/weights/best.onnx", providers=EP_list)
    return sess


def preprocess(img: np.array, bgr2rgb: bool = True) -> np.array:
    img_copy = np.copy(img)
    if bgr2rgb:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    # ig =cv2.resize(img, (640, 640))
    img_copy = np.transpose(img_copy, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_copy = np.expand_dims(img_copy, axis=0)
    img_copy = np.ascontiguousarray(img_copy / 255)
    return img_copy


def run(model: ort.InferenceSession, inp: np.array) -> np.array:
    outputs = model.run(None, {"images": inp})
    return outputs[0][0]


def postprocess(raw_out: np.array) -> np.array:
    correct = []
    for pred in raw_out.T:
        x, y, w, h, score = pred
        if score > 0.5:
            x1 = int(x - w / 2)
            x2 = int(x + w / 2)
            y1 = int(y - h / 2)
            y2 = int(y + h / 2)
            correct.append([x1, y1, x2, y2, score])
    correct = np.array(correct)
    final_boxes = nms(correct)
    return final_boxes


def process(model: ort.InferenceSession, img: np.array) -> np.array:
    inp = preprocess(img)
    out = run(model, inp)
    boxes = postprocess(out)
    return boxes


def draw(img: np.array, boxes: np.array, count: int) -> np.array:
    img_copy = np.copy(img)

    img_copy = cv2.line(img_copy, [200, 380], [380, 380], color=(0, 255, 0), thickness=1)
    img_copy = cv2.line(img_copy, [200, 385], [380, 385], color=(0, 255, 0), thickness=1)
    img_copy = cv2.putText(
        img_copy, f"Count: {count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    for el in boxes:
        x1, y1, x2, y2, person_id = el.astype(np.int32)
        img_copy = cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        img_copy = cv2.circle(
            img_copy, (int(x1 + x2) // 2, int(y2 + y1) // 2), radius=2, color=(255, 0, 0), thickness=2
        )
    return img_copy


def intersect_1(a: int, b: int, c: int, d: int):
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    return max(a, c) <= min(b, d)


def main():
    count = 0
    yolo = init_onnx("./runs/detect/train2/weights/best.onnx")
    cap = cv2.VideoCapture("/storage/data/2023-04-29-05-05-16.mp4")  # video file
    mot_tracker = Sort(max_age=15, iou_threshold=0.3)
    prev_trackers = []  # trackers on current-1 frame
    current_trackers = []  # trackers on current frame
    border_data = {}  # dict of ids of people who crossed the 1st border
    while True:
        ret, frame = cap.read()  # get current frame
        frame = cv2.resize(frame, (640, 640))
        dets = process(yolo, frame)  # get detections

        current_trackers = np.copy(mot_tracker.update(dets))  # update SORT with detections
        # run pairvise on prev and current trackers with same id
        for prev_box, curr_box in filter(lambda x: x[0][-1] == x[1][-1], product(prev_trackers, current_trackers)):
            prev_y = int((prev_box[3] + prev_box[1]) / 2)
            curr_y = int((curr_box[3] + curr_box[1]) / 2)
            fid = int(curr_box[-1])
            if intersect_1(prev_y, curr_y, 380, 380):
                border_data[fid] = True  # if person crosses the 1st border
            if fid in border_data and intersect_1(prev_y, curr_y, 385, 385): 
                if border_data[fid]:
                    count += 1  # increment count if person crosses the 2nd border after 1st border
                    border_data[fid] = False  # reset border_data

        frame = draw(frame, current_trackers, count)  # draw boxes and borders on frame
        cv2.imshow("video feed", frame)

        prev_trackers = np.copy(current_trackers)  # update prev_trackers

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
