import pandas as pd

def read_coordenates_csv(path):
    frames = pd.read_csv(path)
    frames["timestamp"] = pd.to_timedelta(frames["timestamp"], unit="ms")
    frames = frames.set_index("timestamp")
    frames.index = frames.index.rename("timestamp")
    frames["ground_truth"] = "idle"

    # select useful coordinates:
    frames_cropped = frames[["ground_truth"]]
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

frames_path = "./data/generated_csvs/csv_file_validation_sergio_2.csv"
elan_path = "./data/elans/sergio_validation_2.txt"

frames = read_coordenates_csv(frames_path)
change_idle_to_gestures(frames,elan_path)
frames.index = (frames.index / 1000000).astype(int)
frames = frames.reset_index(drop=False)

frames.to_csv("./data/test_mode_outputs/csv_with_ground_truth.csv", index=False)

print("csv_with_ground_truth.csv created")