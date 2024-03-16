from filterpy.kalman import KalmanFilter
import numpy as np

from kalman_trackers.kalman_box_tracker_ds import KalmanBoxTrackerDS


class KalmanBoxTrackerBotSort(KalmanBoxTrackerDS):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        std = [
            (self._std_weight_position * self.kf.x[2]) ** 2,
            (self._std_weight_position * self.kf.x[3]) ** 2,
            (self._std_weight_position * self.kf.x[2]) ** 2,
            (self._std_weight_position * self.kf.x[3]) ** 2,
        ]
        np.fill_diagonal(self.kf.R, np.square(std))
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
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
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def _convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
          [x,y,w,h] where x,y is the centre of the box and w, h is the width and height
          of th box
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        return np.array([x, y, w, h]).reshape((4, 1))

    def _convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,h] and returns it in the form
          [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = x[2]
        h = x[3]
        if score is None:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
