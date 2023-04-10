import numpy as np
import pickle
import py_classes.preprocess as prep

frame_array , movements_array = prep.generate_data_lists("./data/generated_csvs/csv_file_rotate_maria.csv","./data/elans/maria_r.txt")
frame_array2, movements_array2 = prep.generate_data_lists("./data/generated_csvs/csv_file_rotate_pablo.csv","./data/elans/pablo_r.txt")
frame_array3, movements_array3 = prep.generate_data_lists("./data/generated_csvs/csv_file_rotate_sergio.csv","./data/elans/sergio_r.txt")
frame_array4, movements_array4 = prep.generate_data_lists("./data/generated_csvs/csv_file_swipe_left_maria.csv","./data/elans/maria_sl.txt")
frame_array5, movements_array5 = prep.generate_data_lists("./data/generated_csvs/csv_file_swipe_left_pablo.csv","./data/elans/pablo_sl.txt")
frame_array6, movements_array6 = prep.generate_data_lists("./data/generated_csvs/csv_file_swipe_left_sergio.csv","./data/elans/sergio_sl.txt")
frame_array7, movements_array7 = prep.generate_data_lists("./data/generated_csvs/csv_file_swipe_right_maria.csv","./data/elans/maria_sr.txt")
frame_array8, movements_array8 = prep.generate_data_lists("./data/generated_csvs/csv_file_swipe_right_pablo.csv","./data/elans/pablo_sr.txt")
frame_array9, movements_array9 = prep.generate_data_lists("./data/generated_csvs/csv_file_swipe_right_sergio.csv","./data/elans/sergio_sr.txt")
frame_array10, movements_array10 = prep.generate_data_lists("./data/generated_csvs/csv_file_noise_pablo.csv","./data/elans/pablo_noise.txt")
frame_array11, movements_array11 = prep.generate_data_lists("./data/generated_csvs/csv_file_rotate_left_pablo.csv","./data/elans/pablo_rotate_left.txt")
frame_array12, movements_array12 = prep.generate_data_lists("./data/generated_csvs/csv_file_rotate_left_sergio.csv","./data/elans/sergio_rotate_left.txt")
frame_array13, movements_array13 = prep.generate_data_lists("./data/generated_csvs/csv_file_table_flip_pablo.csv","./data/elans/pablo_table_flip.txt")
frame_array14, movements_array14 = prep.generate_data_lists("./data/generated_csvs/csv_file_table_flip_pablo_2.csv","./data/elans/pablo_table_flip_2.txt")
frame_array15, movements_array15 = prep.generate_data_lists("./data/generated_csvs/csv_file_table_flip_sergio.csv","./data/elans/sergio_table_flip.txt")
frame_array16, movements_array16 = prep.generate_data_lists("./data/generated_csvs/csv_file_table_flip_sergio_2.csv","./data/elans/sergio_table_flip_2.txt")
frame_array17, movements_array17 = prep.generate_data_lists("./data/generated_csvs/csv_file_zoom_in_pablo.csv","./data/elans/pablo_zoom_in.txt")
frame_array18, movements_array18 = prep.generate_data_lists("./data/generated_csvs/csv_file_zoom_in_pablo_2.csv","./data/elans/pablo_zoom_in_2.txt")
frame_array19, movements_array19 = prep.generate_data_lists("./data/generated_csvs/csv_file_zoom_in_sergio.csv","./data/elans/sergio_zoom_in.txt")
frame_array20, movements_array20 = prep.generate_data_lists("./data/generated_csvs/csv_file_zoom_out_pablo.csv","./data/elans/pablo_zoom_out.txt")
frame_array21, movements_array21 = prep.generate_data_lists("./data/generated_csvs/csv_file_zoom_out_sergio.csv","./data/elans/sergio_zoom_out.txt")
frame_array22, movements_array22 = prep.generate_data_lists("./data/generated_csvs/csv_file_zoom_out_sergio_2.csv","./data/elans/sergio_zoom_out_2.txt")

X = np.concatenate((frame_array, frame_array2, frame_array3, frame_array4, frame_array5, frame_array6, frame_array7, frame_array8, frame_array9, frame_array10, frame_array11, frame_array12, frame_array13, frame_array14, frame_array15, frame_array16, frame_array17, frame_array18, frame_array19, frame_array20, frame_array21, frame_array22))
Y = np.concatenate((movements_array, movements_array2, movements_array3, movements_array4, movements_array5, movements_array6, movements_array7, movements_array8, movements_array9, movements_array10, movements_array11, movements_array12, movements_array13, movements_array14, movements_array15, movements_array16, movements_array17, movements_array18, movements_array19, movements_array20, movements_array21, movements_array22))

print(X.shape)
print(Y.shape)

with open('./data/XY/X.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('./data/XY/Y.pkl', 'wb') as f:
    pickle.dump(Y, f)