import pickle
import numpy as np
import sys
import os
import py_classes.preprocess_test_mode as prep
from py_classes.movements import Movements

current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file)
framework_directory = os.path.abspath(os.path.join(parent_directory, '../../Framework/'))
sys.path.append(framework_directory)

input_path = "./data/generated_csvs/csv_file_validation_sergio_2.csv"
output_path = "./data/test_mode_outputs/emitted_events.csv"

with open('./data/model/NN.pkl', 'rb') as f:
    NN = pickle.load(f)

with open('./data/XY/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

output_df, resampled_df, frame_list = prep.generate_data_lists(input_path)
x = scaler.transform(np.array(frame_list))

predictions = []

window = 10
index = 39
timeouts = [False] * len(Movements)
to_csv = []

for i, frame in enumerate(x):
    prediction = NN.predict(frame)
    predictions.append(prediction)
    for move in range(1,len(Movements)):
        if predictions[-window:].count(move) >= window * 0.8:
            if timeouts[move-1] != True:
                timeouts[move-1] = True
                
                time_to_change = resampled_df.iloc[index,0]
                diff= abs(output_df["timestamp"] - time_to_change)
                min_idx = diff.idxmin()
                result = output_df.loc[min_idx]
                output_df.loc[min_idx,"events"] = Movements(move).name

        elif predictions[-window:].count(move) < window * 0.2:
            if timeouts[move-1] == True:
                timeouts[move-1] = False

    index +=1

output_df.to_csv(output_path, index=False)
print("emitted_events.csv created")

#python 7_calculator.py --events_csv=data/test_mode_outputs/emitted_events.csv --ground_truth_csv=data/test_mode_outputs/csv_with_ground_truth.csv
