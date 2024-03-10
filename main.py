import os

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from sort import Sort
from utils import init_onnx, process_image

np.random.seed(0)
MOT_FOLDER = Path('./MOT17/train')
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help='experiment name')


def process_video(model, sequence_name: str, results_folder: Path):
    sequence_folder = MOT_FOLDER/sequence_name/'img1'
    frames = sorted(next(os.walk(sequence_folder))[2])
    result_file = results_folder/f'{sequence_name}.txt'
    mot_tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)  # create instance of the SORT tracker

    with open(result_file, 'w') as f:
        for i, frame in enumerate(tqdm(frames, total=len(frames), desc=str(sequence_name))):
            detections = process_image(model, sequence_folder/frame)
            trackers = np.copy(mot_tracker.update(detections))
            for tracker in trackers:
                obj_id = int(tracker[-1])
                bb_left = tracker[0]
                bb_top = tracker[1]
                bb_width = tracker[2] - tracker[0]
                bb_height = tracker[3] - tracker[1]
                f.write(f'{i+1} {obj_id} {bb_left} {bb_top} {bb_width} {bb_height} 1 -1 -1 -1\n')

def main():
    count = 0
    yolo = init_onnx("./weights/mot17-01-frcnn.onnx")

    args = parser.parse_args()
    results_folder = Path(f'./results/{args.name}/data')
    results_folder.mkdir(parents=True, exist_ok=True)

    sequences = next(os.walk(MOT_FOLDER))[1]
    for sequence_name in sequences:
        process_video(yolo, sequence_name, results_folder)





    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.resize(frame, (640, 640))
    #     dets = process_image(yolo, frame)
    #
    #     current_trackers = np.copy(mot_tracker.update(dets))
    #
    #     cv2.imshow("video feed", frame)
    #
    #     prev_trackers = np.copy(current_trackers)
    #
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break



if __name__ == "__main__":
    main()
