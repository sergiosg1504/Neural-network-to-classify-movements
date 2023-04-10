import threading
import time
import cv2
import numpy as np
import pandas as pd 
from py_classes.movements import Movements
from sanic import Sanic
from sanic.response import html
from sanic.response import text
import pickle
import cv2
import mediapipe as mp
import yaml
import pathlib
import multiprocessing

import requests
import asyncio
import os
import sys


current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file)
framework_directory = os.path.abspath(os.path.join(parent_directory, '../../Framework/'))
sys.path.append(framework_directory)


livemode = True
webSocket = None

def change_coordenates_to_differences(frame):
    frame_changed = frame.diff()
    frame_changed = frame_changed.drop(0)
    frame_changed = frame_changed.reset_index(drop=True)
    return frame_changed


def flatten_frame(frame_cutted):
    return frame_cutted.values.flatten()


def function1():

    timeouts = [False] * len(Movements)
    percentaje_needed = [0.8, 0.8, 0.8, 0.8, 0.5, 0.7, 0.5]
    predictions = [0] * 40
    window = 60

    while not end:
        lock_L1.acquire()
        if(len(L1) < 50):
            lock_L1.release()
            time.sleep(0.1)
        else:
            df = pd.DataFrame(L1)
            lock_L1.release()
            
            df = df.rename(columns={0:'timestamp', 1: 'left_shoulder_x', 2: 'left_shoulder_y', 3: 'right_shoulder_x', 4: 'right_shoulder_y', 5: 'left_elbow_x', 6: 'left_elbow_y', 7: 'right_elbow_x', 8: 'right_elbow_y', 9: 'left_wrist_x', 10: 'left_wrist_y', 11: 'right_wrist_x', 12: 'right_wrist_y', 13: 'left_pinky_x', 14: 'left_pinky_y', 15: 'right_pinky_x', 16: 'right_pinky_y', 17: 'left_index_x', 18: 'left_index_y', 19: 'right_index_x', 20: 'right_index_y', 21: 'left_thumb_x', 22: 'left_thumb_y', 23: 'right_thumb_x', 24: 'right_thumb_y'})
            df["timestamp"] = pd.to_timedelta(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            df.index = df.index.rename("timestamp")
            df = df.resample('40ms').mean().interpolate(method='time')

            frames = df[-40:]
            current_frame = frames.reset_index(drop=True)
            current_frame = change_coordenates_to_differences(current_frame)
            frame_flattened = flatten_frame(current_frame)
            frame_flattened = np.array(frame_flattened)

            frame_normalized = scaler.transform(frame_flattened)

            predictions.append(NN.predict(frame_normalized))
            for move in range(1,len(Movements)):
                if predictions[-window:].count(move) >= window * percentaje_needed[move-1]:
                    if timeouts[move-1] != True:
                        timeouts[move-1] = True
                        print(Movements(move).name)
                        requests.get(f"http://127.0.0.1:8000/send?event={Movements(move).name}")
                elif predictions[-window:].count(move) < window * 0.1:
                    if timeouts[move-1] == True:
                        timeouts[move-1] = False

slideshow_root_path = pathlib.Path(__file__).parent.joinpath("slideshow")
print(slideshow_root_path)
app = Sanic(name = "slideshow_server")
app.static("/static", "./slideshow")

@app.route("/")
async def index(request):
    return html(open(slideshow_root_path / "slideshow.html", "r").read())

@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")
    global webSocket
    webSocket = ws
    while True:
        await asyncio.sleep(2)

@app.route("/send")
async def send(request):
    global webSocket
    if webSocket != None:
        event = request.args["event"][0]
        await webSocket.send(event)
        return text(f"Server received: {event}")
    return text(f"webServer is none.")

def initializeServer():
    app.run(host="127.0.0.1", debug=False)



if __name__ == "__main__":

    L1 = []
    lock_L1 = threading.Lock()
    end = False

    with open('./data/model/NN.pkl', 'rb') as f:
        NN = pickle.load(f)

    with open('./data/XY/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    t1 = threading.Thread(target=function1)
    t2 = multiprocessing.Process(target=initializeServer)

    t2.start()
    time.sleep(5)
    t1.start()
    

    if livemode:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        show_data = True
        show_video = True

        cap = cv2.VideoCapture(index=0)

        with open("./video_to_csv_helpers/keypoint_mapping.yml", "r") as yaml_file:
            mappings = yaml.safe_load(yaml_file)
            KEYPOINT_NAMES = mappings["face"]
            KEYPOINT_NAMES += mappings["body"]

        success = True
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and success:
                success, image = cap.read()
                if not success:
                    break
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                if show_video:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow('MediaPipe Pose', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    end = True
                    break

                if show_data and results.pose_landmarks is not None:
                    frame = []
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    frame.append(timestamp)
                    
                    for joint_name in ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb"]: # you can choose any joint listed in `KEYPOINT_NAMES`
                        joint_data = results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)]
                        frame.append(joint_data.x)
                        frame.append(joint_data.y)
                    lock_L1.acquire()
                    L1.append(frame)
                    lock_L1.release()

        cap.release()
        
    t1.join()
    t2.join()


    
