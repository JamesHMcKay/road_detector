import create_training_set
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
import yaml

total_iterations = 0

def train(num_iteration):
    global total_iterations
    for i in range(
            total_iterations,
            total_iterations + num_iteration):

        x_batch, y_true_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch = data.valid.next_batch(batch_size)
        feed_dict_tr = {
            x: x_batch,
            y_true: y_true_batch}
        feed_dict_val = {
            x: x_valid_batch,
            y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            print_status(feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './model/trained-model')
    total_iterations += num_iteration


def print_status(feed_dict_train, feed_dict_validate, validation_loss):
    training_accuracy = session.run(accuracy, feed_dict=feed_dict_train)
    validation_accuracy = session.run(accuracy, feed_dict=feed_dict_validate)
    output = "Training accuracy: {0:>6.1%}, Validation Accuracy: {1:>6.1%},\
        Validation Loss: {2:.3f}"
    print(output.format(
        training_accuracy,
        validation_accuracy,
        validation_loss))

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(
        input,
        num_input_channels,
        conv_filter_size,
        num_filters):
    weights = create_weights(shape=[
            conv_filter_size,
            conv_filter_size,
            num_input_channels,
            num_filters])

    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(
        input=input,
        filter=weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(
        value=layer,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(
        input,
        num_inputs,
        num_outputs,
        use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases
    # use rectified linear activiation function
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


def train_model(file_path):
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    global batch_size
    global num_classes
    global img_size
    global num_channels
    global data

    config = yaml.safe_load(open("config.yaml"))
    num_classes = config['training_parameters']['number_of_classes']
    img_size = config['image_processing']['training_image_size']
    num_channels = config['image_processing']['number_of_channels']
    batch_size = config['training_parameters']['batch_size']

    data = create_training_set.process_images(file_path)

    global session
    session = tf.Session()

    global x
    global y_true
    global y_true_cls
    x = tf.placeholder(
        tf.float32,
        shape=[None, img_size, img_size, num_channels],
        name='x')
    y_true = tf.placeholder(
        tf.float32,
        shape=[None, num_classes],
        name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64
    fc_layer_size = 128

    layer_conv1 = create_convolutional_layer(
        input=x,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1)
    layer_conv2 = create_convolutional_layer(
        input=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2)
    layer_conv3 = create_convolutional_layer(
        input=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3)

    layer_flat = create_flatten_layer(layer_conv3)

    layer_fc1 = create_fc_layer(
            input=layer_flat,
            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
            num_outputs=fc_layer_size,
            use_relu=True)

    layer_fc2 = create_fc_layer(
        input=layer_fc1,
        num_inputs=fc_layer_size,
        num_outputs=num_classes,
        use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

    y_pred_cls = tf.argmax(y_pred, dimension=1)
    session.run(tf.global_variables_initializer())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=layer_fc2,
        labels=y_true)
    global cost
    cost = tf.reduce_mean(cross_entropy)
    global optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    global accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())
    global saver
    saver = tf.train.Saver()

    train(num_iteration=10000)
