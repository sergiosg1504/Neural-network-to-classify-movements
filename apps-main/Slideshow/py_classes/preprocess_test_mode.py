import pandas as pd

def read_coordenates_csv(path):
    frames = pd.read_csv(path)
    frames["timestamp"] = pd.to_timedelta(frames["timestamp"], unit="ms")
    frames = frames.set_index("timestamp")
    frames.index = frames.index.rename("timestamp")
    # select useful coordinates:
    frames_cropped = frames[["left_shoulder_x","left_shoulder_y","right_shoulder_x","right_shoulder_y","left_elbow_x","left_elbow_y","right_elbow_x","right_elbow_y","left_wrist_x","left_wrist_y","right_wrist_x","right_wrist_y",
        "left_pinky_x","left_pinky_y","right_pinky_x","right_pinky_y","left_index_x","left_index_y","right_index_x","right_index_y","left_thumb_x","left_thumb_y","right_thumb_x","right_thumb_y"]]
    return frames_cropped

def change_coordenates_to_differences(frame):
    frame_changed = frame.diff()
    frame_changed = frame_changed.drop(0)
    frame_changed = frame_changed.reset_index(drop=True)
    return frame_changed

def flatten_frame(frame_cutted):
    return frame_cutted.values.flatten()

def generate_data_lists(frames_path):
    frames = read_coordenates_csv(frames_path)
    output_df = frames.index.astype(int)
    output_df = output_df / 1000000
    output_df = pd.DataFrame(output_df).astype(int)
    output_df = output_df.assign(events="idle")

    frames = frames.resample('40ms').mean().interpolate(method='time')

    resampled_df = frames.index.astype(int)
    resampled_df = resampled_df / 1000000
    resampled_df = pd.DataFrame(resampled_df).astype(int)
    resampled_df = resampled_df.assign(prediction=None)

    frame_list = []
    start_frame = window = 40             # 25 fps = 40 ms
    frames = frames.reset_index(drop=True)
    while start_frame < len(frames):
        current_frame = frames.iloc[start_frame - window : start_frame]
        current_frame = current_frame.reset_index(drop=True)
        current_frame = change_coordenates_to_differences(current_frame)
        frame_flattened = flatten_frame(current_frame)
        frame_list.append(frame_flattened)
        start_frame += 1
    return output_df, resampled_df, frame_list