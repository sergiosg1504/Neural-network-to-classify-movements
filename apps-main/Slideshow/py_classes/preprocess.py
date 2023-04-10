import pandas as pd
import numpy as np
import pickle
import sys
from .movements import Movements
import os

current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file)
framework_directory = os.path.abspath(os.path.join(parent_directory, '../../../Framework/'))

sys.path.append(framework_directory)
from standardScaler import StandardScaler
import utils


def read_coordenates_csv(path):
    frames = pd.read_csv(path)
    frames["timestamp"] = pd.to_timedelta(frames["timestamp"], unit="ms")
    frames = frames.set_index("timestamp")
    frames.index = frames.index.rename("timestamp")
    frames = frames.resample('40ms').mean().interpolate(method='time')
    frames["ground_truth"] = "idle"

    # select useful coordinates:
    frames_cropped = frames[["left_shoulder_x","left_shoulder_y","right_shoulder_x","right_shoulder_y","left_elbow_x","left_elbow_y","right_elbow_x","right_elbow_y","left_wrist_x","left_wrist_y","right_wrist_x","right_wrist_y",
        "left_pinky_x","left_pinky_y","right_pinky_x","right_pinky_y","left_index_x","left_index_y","right_index_x","right_index_y","left_thumb_x","left_thumb_y","right_thumb_x","right_thumb_y","ground_truth"]]
    return frames_cropped

def read_elan(path):
    annotations = pd.read_csv(path, sep="\t", header=None, usecols=[3, 5, 8], names=["start", "end", "label"])
    annotations["start"] = pd.to_timedelta(annotations["start"], unit="s")
    annotations["end"]   = pd.to_timedelta(annotations["end"], unit="s")
    return annotations

def change_idle_to_gestures(frames, path):

    annotations = read_elan(path)

    for idx, ann in annotations.iterrows():
        if ann["label"] != "nothing":
            annotated_frames = (frames.index >= ann["start"]) & (frames.index <= ann["end"])
            frames.loc[annotated_frames, "ground_truth"] = ann["label"]

def change_coordenates_to_differences(frame):
    frame_changed = frame.iloc[:, :-1].diff()
    frame_changed = frame_changed.drop(0)
    frame_changed = frame_changed.reset_index(drop=True)
    return frame_changed

def flatten_frame(frame_cutted):
    return frame_cutted.values.flatten()

def generate_data_lists(frames_path, elan_path):

    frames = read_coordenates_csv(frames_path)
    change_idle_to_gestures(frames,elan_path)

    frame_list = []
    movements_list = []

    start_frame = window = 40             # 25 fps = 40 ms

    frames = frames.reset_index(drop=True)
    while start_frame < len(frames):
        move = frames.iloc[start_frame, len(frames.iloc[0]) -1]
        move = Movements[move].value
        movements_list.append(move)

        current_frame = frames.iloc[start_frame - window : start_frame]
        current_frame = current_frame.reset_index(drop=True)
        current_frame = change_coordenates_to_differences(current_frame)
        frame_flattened = flatten_frame(current_frame)
        frame_list.append(frame_flattened)
        start_frame += 1
    return frame_list, movements_list

def loadXY():
    with open('./data/XY/X.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('./data/XY/Y.pkl', 'rb') as f:
        Y = pickle.load(f)

    perm = np.random.permutation(X.shape[0])
    X = X[perm].astype(float)
    Y = Y[perm]
    
    NUM_TRAINING_SAMPLES = (int)(len(X) * 0.9)

    x_train = X[:NUM_TRAINING_SAMPLES]
    x_validation  = X[NUM_TRAINING_SAMPLES:]

    scaler = StandardScaler()
    scaler.fit(x_train) # fit once on the training data

    x_train = scaler.transform(x_train)
    x_validation = scaler.transform(x_validation)

    Y_hot = utils.one_hot(Y)
    
    y_train = Y_hot[:NUM_TRAINING_SAMPLES]
    y_validation  = Y_hot[NUM_TRAINING_SAMPLES:]

    saveXY(x_train, y_train, x_validation, y_validation, scaler)

    return x_train, y_train, x_validation, y_validation, scaler

def saveXY(x_train, y_train, x_validation, y_validation, scaler):
    with open('./data/XY/X_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)

    with open('./data/XY/Y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

    with open('./data/XY/X_validation.pkl', 'wb') as f:
        pickle.dump(x_validation, f)

    with open('./data/XY/Y_validation.pkl', 'wb') as f:
        pickle.dump(y_validation, f)

    with open('./data/XY/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def loadXY_validation():

    with open('./data/XY/X_train.pkl', 'rb') as f:
        x_train = pickle.load(f)

    with open('./data/XY/Y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open('./data/XY/X_validation.pkl', 'rb') as f:
        x_validation = pickle.load(f)

    with open('./data/XY/Y_validation.pkl', 'rb') as f:
        y_validation = pickle.load(f)

    return x_train, y_train, x_validation, y_validation