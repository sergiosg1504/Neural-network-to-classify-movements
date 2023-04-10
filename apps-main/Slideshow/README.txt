######################################################
#    Pablo Santos Blázquez - mat. number: 2878274    #
#    Sergio Sánchez García - mat. number:            #
######################################################
#     Controlling slideshow server via gestures      #
######################################################

· Folder architecture:

Slideshow/
├─ 1_video_to_csv.py
├─ 2_preprocess.py
├─ 3_training.py
├─ 4_validation.ipynb
├─ 5_live_mode.py
├─ 6_1_ground_truth_csv_generator.py
├─ 6_2_testmode.py
├─ 7_calculator.py
├─ data/
│  ├─ ...
├─ py_classes/
│  ├─ ...
├─ slideshow/
│  ├─ ...
├─ video_to_csv_helpers/
│  ├─ ...
├─ README.txt

Install the required packets listed in data/requirements.txt with the following command:

pip install -r data/requirements.txt

- 1_video_to_csv.py: parse videos to a CSV file extracting coordinates of the user's joints.
- 2_preprocess.py: processes the data to fit the input of the neural network.
- 3_training.py: contains the code for training the neural network.
- 4_validation.ipynb: notebook to visualize the accuracy, error, F1 score and confusion matrix.
- 5_live_mode.py: executes the slideshow server at localhost:8000 and uses the neural network to control it via gestures.
- 6_1_ground_truth_csv_generator.py: generates a CSV file with the ground truth.
- 6_2_testmode.py: generates a CSV file with the events predicted by the neural network.
- 7_calculator.py: compute the performance score of the neural network by using the previous CSV files.
- data/: contains all the data created and/or needed during the above steps.
- py_classes/: contains the implementation of auxiliar functions used on the previous files.
- README.txt: file explaning folder architecture.