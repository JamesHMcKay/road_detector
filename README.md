[![Build Status](https://travis-ci.org/JamesHMcKay/road_detector.svg?branch=master)](https://travis-ci.org/JamesHMcKay/road_detector)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Introduction

This package extracts a road network from satellite images using computer vision techniques.

## Prerequisites

See requirements.txt for the required Python packages.

## Instructions

The trained model is included in this repository, so if desired skip immediately to "Testing the model".  The training should take about half an hour to an hour on a desktop computer, and testing of the trained model should take no more than minute or two per image.

### Training the model

To train the model run

python road_detector/road_detector.py train PATH

where PATH is the location of the training data.  This location must contain the .jpg satellite images, and .tif files.
The .tif files are binary representations of the road network (with values between 0 and 1).

### Testing the model

To test the model run

python road_detector/road_detector.py test PATH

where PATH is the location of the test images.  This will output the resultant raw and processed predictions to the root directory.

### Testing a single image

python road_detector/road_detector.py test-single PATH

where PATH is the location of the test image.  This will output the resultant raw and processed predictions to the root directory.

## Advanced use

The config.yaml file located in the root directory can be modified to allow for different input types and different settings for the training and testing of the model.

It is possible to perform more specific actions, such as only post-processing, or testing the trained network on a single image instead of an entire directory. See road_detector/road_detector.py for more information.

## Tests

The test suite is currently under-development and is no where near complete.
