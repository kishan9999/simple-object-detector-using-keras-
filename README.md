# Simple object detector using keras
very basic version of custom object detection using pre trained model.

-Model can be trained using differenet pre trained model.
-coordinate normlisation has been performed
-Special relu activation is used instead of softmax for object detection
-lower learning  rate is choosed

# Implementation
step 1: annote datasets (images) labelImg.exe software
step 2: Process data sets using data_preperations.py
Step 3: run train_detector2.py 

# Datasets Format:
Custom datasets created to detect star symbol with different size and locations. 

# Major Python-Libraries used:
1. tensorflow 2.3.0
2. keras-tuner: 1.0.1
3. matplotlib: 3.2.2
4. pandas: 1.0.5
5. numpy: 1.18.5

# References:
https://keras.io/api/applications/inceptionv3/
https://github.com/Samjith888/Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-
