"""
Replays Training Data
"""

import csv
import cv2

DRIVING_LOG_FILE = 'data/driving_log.csv'
IMAGE_DATA_DIR = 'data/IMG/'

with open(DRIVING_LOG_FILE) as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:

        # discard csv header line
        if line[0] == 'center':
            continue

        center_image = cv2.imread(IMAGE_DATA_DIR + line[0].split('/')[-1])
        left_image = cv2.imread(IMAGE_DATA_DIR + line[1].split('/')[-1])
        right_image = cv2.imread(IMAGE_DATA_DIR + line[2].split('/')[-1])

        cv2.namedWindow("LeftCamera")
        cv2.namedWindow("CenterCamera")
        cv2.namedWindow("RightCamera")

        cv2.imshow("LeftCamera", left_image)
        cv2.imshow("CenterCamera", center_image)
        cv2.imshow("RightCamera", right_image)

        cv2.waitKey(25)

    cv2.waitKey(-1)
