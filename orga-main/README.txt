######################################################
#    Pablo Santos Blázquez - mat. number: 2878274    #
#    Sergio Sánchez García - mat. number: 2861907    #
######################################################
#         Architecture of our final project          #
######################################################

The project is distributed in 2 repositories:

Final Project/
├─ Apps/
│  ├─ MNIST_example/
│  ├─ Slideshow/
│  ├─ README.txt
│  ├─ ...
├─ Framework/
│  ├─ README.txt
│  ├─ ...

1) Apps:
In this repository you can find 2 examples in which we use our Neural Network Framework:
   - MNIST_example/: this folder contains the necessary files to demonstrate the usage of our framework with the MNIST example. (See its README.txt for further details)
   - Slideshow/: this folder contains the data and code necessary to control a slideshow via gestures. (See its README.txt for further details)

2) Framework:
In this repository you can find all the necessary code to create a Neural Network form scratch. (See its README.txt for further details)

In order to download and setup both repositories the next steps must be followed:
   · 1) Download framework repository: git clone https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-14/framework.git
   · 2) Download Apps: git clone https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-14/apps.git
   · 3) Rename framework to Framework.
   · 4) Rename apps to Apps. 

If you are using Windows as your OS you can just execute: setup.bat.
This file will do that work for you.

The final folder should look like this:

orga/
├─ Apps/
│  ├─ MNIST_example/
│  ├─ Slideshow/
│  ├─ README.txt
│  ├─ ...
├─ Framework/
│  ├─ README.txt
│  ├─ ...
├─ .gitignore
├─ Presentation.pptx
├─ Presentation_video.mp4
├─ README.txt
├─ setup.bat


Selected optional requirements:
1) O1: Presentation of your approaches towards data preparation, hyperparameter choices and result evaluations [10 points].
2) O2: Machine Learning Framework Package [15 points].
3) O3: rotate left/right [5 points].
4) O4: pinch/spread [5 points].
5) O6: flip table (both hands up) [5 points].
