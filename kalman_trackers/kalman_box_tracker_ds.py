from filterpy.kalman import KalmanFilter
import numpy as np

from kalman_trackers.base_kalman_box_tracker import KalmanBoxTracker


class KalmanBoxTrackerDS(KalmanBoxTracker):

    def _init_kf(self, bbox):
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

    def update(self, bbox):
        self.time_since_update = 0
        self.trek = []
        std = [
            self._std_weight_position * self.kf.x_prior[3],
            self._std_weight_position * self.kf.x_prior[3],
            np.array([1e-1]),
            self._std_weight_position * self.kf.x_prior[3]]
        np.fill_diagonal(self.kf.R, np.square(std))
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        std = [
            2 * self._std_weight_position * self.kf.x[3],
            2 * self._std_weight_position * self.kf.x[3],
            np.array([1e-2]),
            2 * self._std_weight_position * self.kf.x[3],
            10 * self._std_weight_velocity * self.kf.x[3],
            10 * self._std_weight_velocity * self.kf.x[3],
            np.array([1e-5]),
            10 * self._std_weight_velocity * self.kf.x[3]]
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
        s = w * h
        return np.array([x, y, s, h]).reshape((4, 1))

    def _convert_x_to_bbox(self, x, score=None):
        w = x[2]/x[3]
        h = x[3]
        if score is None:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
