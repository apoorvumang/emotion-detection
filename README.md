emotion-detection
=================

Predict interest level of user based on facial expressions.

To compile:
1. Packages required: libopencv-dev, build-essential
2. After installing packages, simply run make

To run:
./main will run the main program that detects user interest level. Press <space> to stop the program and get the average interest level.

To train:
For each facial feature, a <feature>-train and <feature>-detect program has been given. These can be used to train these features as well as test their detection independently from the main program.
After training, a csv file of the trained data will have to be generated using the create_csv.py script provided