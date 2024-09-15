from kalman_trackers.base_kalman_box_tracker import KalmanBoxTracker
from kalman_trackers.eight_component_kalman_box_tracker import KalmanBoxTrackerWH, KalmanBoxTrackerSH
from kalman_trackers.kalman_box_tracker_botsort import KalmanBoxTrackerBotSort
from kalman_trackers.kalman_box_tracker_ds import KalmanBoxTrackerDS

KALMAN_BOX_TRACKER = KalmanBoxTracker

NMS_THRESH = 0.5
INP_SIZE = (640, 640)
