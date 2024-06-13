import numpy as np

from config import KALMAN_BOX_TRACKER
from utils import gen_iou_matrix, linear_assignment


def detection2tracker_associate(model_detections, current_trackers, iou_thresh=0.3):
    if len(current_trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(model_detections)), np.empty((0, 5), dtype=int)

    iou_matrix = gen_iou_matrix(model_detections, current_trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_thresh).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_ids = np.stack(np.where(a), axis=1)
        else:
            matched_ids = linear_assignment(-iou_matrix)
    else:
        matched_ids = np.empty(shape=(0, 2))

    unmatched_dets = []
    for d, det in enumerate(model_detections):
        if d not in matched_ids[:, 0]:
            unmatched_dets.append(d)
    unmatched_trks = []
    for t, trk in enumerate(current_trackers):
        if t not in matched_ids[:, 1]:
            unmatched_trks.append(t)

    filtered_matches = []
    for m in matched_ids:
        if iou_matrix[m[0], m[1]] < iou_thresh:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            filtered_matches.append(m.reshape(1, 2))
    if len(filtered_matches) == 0:
        filtered_matches = np.empty((0, 2), dtype=int)
    else:
        filtered_matches = np.concatenate(filtered_matches, axis=0)

    return filtered_matches, np.array(unmatched_dets), np.array(unmatched_trks)


class Sort(object):
    def __init__(self, max_age=1, iou_thresh=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_thresh
        self.current_trackers = []
        self.frame_number = 0

    def update(self, modeldets=np.empty((0, 5))):

        self.frame_number += 1

        tmp_trackers = np.zeros((len(self.current_trackers), 5))
        trackers_to_delete = []
        final_results = []
        for t, tracker in enumerate(tmp_trackers):
            coords = self.current_trackers[t].predict()[0]
            tracker[:] = [coords[0], coords[1], coords[2], coords[3], 0]
            if np.any(np.isnan(coords)):
                trackers_to_delete.append(t)
        tmp_trackers = np.ma.compress_rows(np.ma.masked_invalid(tmp_trackers))
        for t in reversed(
            trackers_to_delete
        ):  # проходим в обратную сторону так как при удалении из списка элемента индексы последующих меняются
            self.current_trackers.pop(t)
        matched_group, unmatched_dets_group, unmatched_trks_group = detection2tracker_associate(
            modeldets,
            tmp_trackers,
            self.iou_threshold
        )

        for m in matched_group:
            self.current_trackers[m[1]].update(modeldets[m[0], :])

        for i in unmatched_dets_group:
            new_tracker = KALMAN_BOX_TRACKER(modeldets[i, :])
            self.current_trackers.append(new_tracker)
        i = len(self.current_trackers)
        for tracker in reversed(self.current_trackers):
            d = tracker.get_state()[0]
            if tracker.time_since_update < 1:
                final_results.append(np.concatenate((d, [tracker.id + 1])).reshape(1, -1))
            i -= 1
            if tracker.time_since_update > self.max_age:
                self.current_trackers.pop(i)
        if len(final_results) > 0:
            return np.concatenate(final_results)
        return np.empty((0, 5))
