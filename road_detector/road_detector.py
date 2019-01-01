import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
import argparse
import create_training_set
import train
import yaml

print('Starting road detector')

if (len(sys.argv) == 3):
    if (sys.argv[1] == "train"):
        train.train_model(sys.argv[2])
    elif (sys.argv[1] == "test"):
        create_training_set.test()
else:
    print('Please provide a command and a path')
