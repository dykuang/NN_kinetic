## This project attempts to construct a 1d convolutional neural network (CNN) for predicting kinetic triplet during thermal analysis.

## Dependency
`Keras > 2.0.0`, `Tensorflow > 1.4.0`

## Other info

* `dataset\`: Contains generated data and some data from real experiment.
* `utils\`: Contains matlab scripts for preprocessing experiment data and some classical regression methods for the prediction
* `datagen_ver2.py`: Generation of data
* `nnkinetic.py`: Main script for constructing, training and testing the network.
* `nnkinetic_cv.py`: Cross validation for a fixed architecture.
* `pre_kinetic.h5`: A pretrained model
* `cvplot.py`: Visualization of results from cross validation. 

