import pathlib
import cv2
import mediapipe as mp
from video_to_csv_helpers import data_to_csv as dtc
import time

# This script uses mediapipe to parse videos to extract coordinates of
# the user's joints. You find documentation about mediapipe here:
#  https://google.github.io/mediapipe/solutions/pose.html

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

# ===========================================================
# ======================= SETTINGS ==========================
show_video = True

cap = cv2.VideoCapture("C:/Users/Pablo/Videos/ML/sergio_validation_2.mp4")  # Video
#cap = cv2.VideoCapture("./data/videos/validation/v1_sergio.mp4")  # Video
#cap = cv2.VideoCapture(index=0) # Live from camera (change index if you have more than one camera)

#result_csv_filename = f"./data/generated_csvs/csv_file_pablo_zoom_out_{current_time}.csv"
result_csv_filename = f"./data/generated_csvs/csv_file_validation_sergio_2.csv"
# ===========================================================

csv_writer = dtc.CSVDataWriter()
success = True
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and success:
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        if show_video:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        # process data
        csv_writer.read_data(data=results.pose_landmarks, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC))

csv_writer.to_csv(result_csv_filename)
cap.release()
