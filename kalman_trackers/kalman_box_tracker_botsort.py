import numpy as np

from kalman_trackers.kalman_box_tracker_ds import KalmanBoxTrackerDS


class KalmanBoxTrackerBotSort(KalmanBoxTrackerDS):

    def update(self, bbox):

        self.time_since_update = 0
        self.trek = []
        std = [
            (self._std_weight_position * self.kf.x[2]) ** 2,
            (self._std_weight_position * self.kf.x[3]) ** 2,
            (self._std_weight_position * self.kf.x[2]) ** 2,
            (self._std_weight_position * self.kf.x[3]) ** 2,
        ]
        np.fill_diagonal(self.kf.R, np.square(std))
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        std = [
            (self._std_weight_position * self.kf.x[2]) ** 2,
            (self._std_weight_position * self.kf.x[3]) ** 2,
            (self._std_weight_position * self.kf.x[2]) ** 2,
            (self._std_weight_position * self.kf.x[3]) ** 2,
            (self._std_weight_velocity * self.kf.x[2]) ** 2,
            (self._std_weight_velocity * self.kf.x[3]) ** 2,
            (self._std_weight_velocity * self.kf.x[2]) ** 2,
            (self._std_weight_velocity * self.kf.x[3]) ** 2,
        ]
        np.fill_diagonal(self.kf.Q, np.square(std))
        self.kf.predict()
        self.time_since_update += 1
        self.trek.append(self._convert_x_to_bbox(self.kf.x))
        return self.trek[-1]

    def _convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        return np.array([x, y, w, h]).reshape((4, 1))

    def _convert_x_to_bbox(self, x, score=None):
        w = x[2]
        h = x[3]
        if score is None:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
