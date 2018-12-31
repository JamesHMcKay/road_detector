import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys,argparse
import create_training_set
from sklearn.utils import shuffle
import create_training_set
import itertools


def classify_image(filename):
    # divide image up into regions and classify each region
    image_size = 100
    img = create_training_set.read_image(filename, image_size)
    result = np.empty([img.shape[0], img.shape[1]])
    verticalDivisor = 12
    horizontalDivisor = 8

    height, width, channels = img.shape
    h = (height / verticalDivisor)
    w = (width / horizontalDivisor)
    quarter_range = int(round(0.25 * h))
    quarter_rangex = int(round(0.25 * w))
    images = []
    number_of_steps = 5
    pixels_per_step_h = int(round(h / number_of_steps))
    pixels_per_step_w = int(round(w / number_of_steps))
    pps_h_2 = int(round(0.5 * pixels_per_step_h))
    pps_w_2 = int(round(0.5 * pixels_per_step_w))

    for i in range(number_of_steps):
        for y in range(verticalDivisor):
            for j in range(number_of_steps):
                for x in range(horizontalDivisor):

                    lower_y = h * y + i * pixels_per_step_h;
                    upper_y = lower_y + h;

                    lower_x = w * x + j * pixels_per_step_w;
                    upper_x = lower_x + w;

                    if (upper_y < height and upper_x < width):
                        subimage = img[lower_y:upper_y, lower_x:upper_x]
                        images.append(create_training_set.convert_image(subimage))

    results = predict(images)

    index = 0
    for i in range(number_of_steps):
        for y in range(verticalDivisor):
            for j in range(number_of_steps):
                for x in range(horizontalDivisor):

                    lower_y = h * y + i * pixels_per_step_h;
                    upper_y = lower_y + h;

                    lower_x = w * x + j * pixels_per_step_w;
                    upper_x = lower_x + w;
                    if (upper_y < height and upper_x < width):
                        likelihood = results[index]
                        like_road = likelihood[0]
                        # if (like_road > 0.9):
                        #     like_road = 1
                        # else:
                        #     like_road = 0
                        index += 1
                        middle_y = int(round(0.5*(upper_y + lower_y)))
                        middle_x = int(round(0.5*(upper_x + lower_x)))
                        result[middle_y - pps_w_2:middle_y + pps_w_2,
                            middle_x-pps_h_2:middle_x + pps_h_2] = like_road*255
    cv2.imwrite("result.jpg", result)


def predict(images):
    # image = create_training_set.convert_image(image)
    # images = []
    image_size = 100
    num_channels = 3
    images = np.array(images)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    
    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('trained-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_empty = np.zeros((1, 2))) 

    max_match_size = 1000
    results = np.empty([1,2])
    # TODO heaps of repeated code below, clean this up
    if (len(images) > max_match_size):
        lower_index = 0
        upper_index = max_match_size
        while (upper_index < len(images)):
            subset = images[lower_index:upper_index]
            x_batch = subset.reshape(subset.shape[0], image_size,image_size,num_channels)
            input_dict = {x: x_batch, y_true: y_empty}
            result=sess.run(y_pred, feed_dict=input_dict)
            results = np.vstack((results, result))
            print('shape of results = ', results.shape)
            lower_index = lower_index + max_match_size
            upper_index = min(upper_index + max_match_size, len(images))
            print(upper_index)
        subset = images[lower_index:upper_index]
        x_batch = subset.reshape(subset.shape[0], image_size,image_size,num_channels)
        input_dict = {x: x_batch, y_true: y_empty}
        result=sess.run(y_pred, feed_dict=input_dict)
        results = np.vstack((results, result))
    else:
        x_batch = images.reshape(images.shape[0], image_size,image_size,num_channels)
        input_dict = {x: x_batch, y_true: y_test_images}
        results=sess.run(y_pred, feed_dict=input_dict)
    return results

classify_image(sys.argv[1])