import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
import argparse
import create_training_set
from sklearn.utils import shuffle
import create_training_set
import itertools
import image_splitter
import yaml


def classify_image(filename):
    # divide image up into regions and classify each region

    config = yaml.safe_load(open("config.yaml"))
    horizontalDivisor = config['image_processing']['horizontal_divisions']
    verticalDivisor = config['image_processing']['vertical_divisions']
    image_size = config['image_processing']['training_image_size']
    number_of_steps = config['image_classification']['number_of_translations']
    img = create_training_set.read_image(
        filename,
        image_size,
        horizontalDivisor,
        verticalDivisor)
    img = create_training_set.convert_image(img)
    result = np.ones([img.shape[0], img.shape[1]])
    result = result * 255
    images = image_splitter.get_sub_images(
        img,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps)

    values = predict(images, image_size)

    result = image_splitter.assign_to_subimages(
        values,
        result,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps)
    cv2.imwrite("result.jpg", result)
    return result


def predict(images, image_size):
    num_channels = 3
    images = np.array(images)
    sess = tf.Session()
    saver = tf.train.import_meta_graph('model/trained-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))

    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_empty = np.zeros((1, 2))

    max_match_size = 1000
    results = np.empty([1, 2])
    if (len(images) > max_match_size):
        lower_index = 0
        upper_index = max_match_size
        while (upper_index < len(images)):
            subset = images[lower_index:upper_index]
            x_batch = subset.reshape(
                subset.shape[0],
                image_size,
                image_size,
                num_channels)
            input_dict = {x: x_batch, y_true: y_empty}
            result = sess.run(y_pred, feed_dict=input_dict)
            results = np.vstack((results, result))
            lower_index = lower_index + max_match_size
            upper_index = min(upper_index + max_match_size, len(images))
        subset = images[lower_index:upper_index]
        x_batch = subset.reshape(
            subset.shape[0],
            image_size,
            image_size,
            num_channels)
        input_dict = {x: x_batch, y_true: y_empty}
        result = sess.run(y_pred, feed_dict=input_dict)
        results = np.vstack((results, result))
    else:
        x_batch = images.reshape(
            images.shape[0],
            image_size,
            image_size,
            num_channels)
        input_dict = {x: x_batch, y_true: y_empty}
        results = sess.run(y_pred, feed_dict=input_dict)
    return results
