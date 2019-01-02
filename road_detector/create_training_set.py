import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
import argparse
import create_training_set
from sklearn.utils import shuffle
import imutils
import re
import image_splitter
import yaml


def test():
    print('test')


class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def process_images(image_size, file_path='../RoadDetection_Train_Images/'):
    global verticalDivisor
    global horizontalDivisor
    global number_of_steps
    config = yaml.safe_load(open("config.yaml"))
    horizontalDivisor = config['image_processing']['horizontal_divisions']
    verticalDivisor = config['image_processing']['vertical_divisions']
    image_size = config['image_processing']['training_image_size']
    number_of_steps = config['image_processing']['number_of_translations']

    image_path = file_path

    class DataSets(object):
        pass
    data_sets = DataSets()
    images = []
    road_images = []
    labels = []

    files = os.listdir(image_path)
    files = list(filter(lambda x: 'jpg' in x and 'aux' not in x, files))
    filenames = list(map((lambda x: re.sub('\.jpg$', '', x)), files))

    for filename in filenames:
        print('reading image ', filename)
        annotated_filename = image_path + filename + ".tif"
        original_filename = image_path + filename + ".jpg"
        image = read_image(
            original_filename,
            image_size,
            horizontalDivisor,
            verticalDivisor)
        roads = read_image(
            annotated_filename,
            image_size,
            horizontalDivisor,
            verticalDivisor)
        # get array of each type from subdivide
        sub_images, sub_lables, sub_roads = subdivide(
            image,
            roads,
            horizontalDivisor,
            verticalDivisor)
        images = images + sub_images
        road_images = road_images + sub_roads
        labels = labels + sub_lables

    images = np.array(images)
    labels = np.array(labels)

    number_of_roads = 0
    for label in labels:
        number_of_roads += int(label[1])
    print('Number of roads in training set = ', number_of_roads)
    number_of_others = int(len(labels) - number_of_roads)
    print('Number of others in training set = ', number_of_others)
    images, labels, road_images = shuffle(images, labels, road_images)
    # index = 0
    # others_count = 0
    # mask = np.full(labels.shape[0], True, dtype=bool)
    # for index in range(labels.shape[0] - 1):
    #     label = labels[index]
    #     if int(label[0]) == 1 and others_count > number_of_roads:
    #         mask[index] = False
    #     elif(int(label[0]) == 1):
    #         others_count += 1

    # labels = labels[mask]
    # images = images[mask]
    # number_of_roads = 0
    # number_of_others = 0
    # for label in labels:
    #     number_of_roads += int(label[1])
    #     number_of_others += int(label[0])


    # print('Number of images = ', labels.shape[0])
    # print('Number of images = ', len(labels))
    # print('Number of roads in training set = ', number_of_roads)
    # print('Number of others in training set = ', number_of_others)
    # for index in range(labels.shape[0] - 1):
    #     label = labels[index]
    #     if int(label[1]) == 1:
    #         cv2.imwrite('roads/road_' + str(index) + '.jpg', images[index])
    #         cv2.imwrite('roads/road_' + str(index) + '_true.jpg', road_images[index])
    #     else:
    #         cv2.imwrite('others/other_' + str(index) + '.jpg', images[index])
    #         cv2.imwrite('others/other_' + str(index) + '_true.jpg', road_images[index])

    # for index in range(images.shape[0]):
    #     images[index] = create_training_set.convert_image(images[index])

    # 20% of the data will automatically be used for validation
    validation_size = 0.2
    validation_size = int(validation_size * images.shape[0])
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.valid = DataSet(validation_images, validation_labels)
    return data_sets


def subdivide(img, img_roads, horizontalDivisor, verticalDivisor):
    images = []
    labels = []

    sub_images = image_splitter.get_sub_images(
        img,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps)
    sub_images_roads = image_splitter.get_sub_images(
        img_roads,
        verticalDivisor,
        horizontalDivisor,
        number_of_steps)

    height = img.shape[0]
    width = img.shape[1]
    h = (height / verticalDivisor)
    w = (width / horizontalDivisor)
    lx = int(round(0.4 * h))
    ux = int(round(0.6 * h))
    ly = int(round(0.4 * w))
    uy = int(round(0.6 * w))
    images_roads = []
    for index in range(len(sub_images) - 1):
        subimage = create_training_set.convert_image(sub_images[index])
        subimage_roads = sub_images_roads[index]
        if (int(np.min(subimage_roads[lx:ux, ly:uy]))) == 0:
            images.append(subimage)
            labels.append([0, 1.0])
            images.append(imutils.rotate(subimage, 90))
            labels.append([0, 1.0])
            images_roads.append(subimage_roads * 255)
            images_roads.append(subimage_roads * 255)
        if (np.min(subimage_roads)) > 0:
            images.append(subimage)
            labels.append([1.0, 0])
            images_roads.append(subimage_roads * 255)
    return images, labels, images_roads


def convert_image(image):
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image

def read_image(filename, image_size, horizontalDivisor, verticalDivisor):
    image = cv2.imread(filename)
    # resize here such that the processing is faster, rather than down
    # size it
    h = image_size * horizontalDivisor
    v = image_size * verticalDivisor
    image = cv2.resize(image, (h, v), 0, 0, cv2.INTER_LINEAR)
    return image
