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

road_count = 0
other_count = 0

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
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]

def process_images(image_size, file_path = '../RoadDetection_Train_Images/'):
    image_path = file_path
    class DataSets(object):
        pass
    data_sets = DataSets()
    images = []
    labels = []

    files = os.listdir(image_path)
    files = list(filter(lambda x: 'jpg' in x and 'aux' not in x, files))
    filenames = list( map( (lambda x: re.sub('\.jpg$', '', x)), files))

    for filename in filenames:
    # for i in range(0,1):
        # filename = filenames[i]
        print('reading image ', filename);
        annotated_filename = image_path + filename + ".tif"
        original_filename = image_path + filename + ".jpg"
        image = read_image(original_filename, image_size)
        roads = read_image(annotated_filename, image_size)
        # get array of each type from subdivide
        sub_images, sub_lables = subdivide(image, roads, 8, 12)
        images = images + sub_images
        labels = labels + sub_lables

    images = np.array(images)
    labels = np.array(labels)

    images, labels = shuffle(images, labels)
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
    global road_count
    global other_count
    images = []
    labels = []
    height, width, channels = img.shape
    h = (height / verticalDivisor)
    w = (width / horizontalDivisor )
    for y in range(verticalDivisor):
        for x in range(horizontalDivisor):
            x2 = width/horizontalDivisor * x
            y2 = height/verticalDivisor * y
            subimage = img[y2:y2+h, x2:x2+w]
            subimage_roads = img_roads[y2:y2+h, x2:x2+w]
            isRoad = False;
            lx = int(round(0.4 * h))
            ux = int(round(0.6 * h))
            ly = int(round(0.4 * w))
            uy = int(round(0.6 * w))
            if (np.min(subimage_roads[lx:ux, ly:uy])) == 0:
                isRoad = True;
            if (isRoad):
                # cv2.imwrite('roads/' + str(road_count) + 'withRoad.jpg', subimage_roads * 255)
                # cv2.imwrite('roads/' + str(road_count) + 'withCroppedRoad.jpg', subimage_roads[lx:ux, ly:uy] * 255)
                # cv2.imwrite('roads/' + str(road_count) + '.jpg', subimage)
                road_count = road_count + 1
                images.append(convert_image(subimage))
                labels.append([0, 1.0])
                images.append(convert_image(imutils.rotate(subimage, 90)))
                labels.append([0, 1.0])
            else:
                # cv2.imwrite('others/' + str(other_count) + '.jpg', subimage)
                other_count = other_count + 1
                images.append(convert_image(subimage))
                labels.append([1.0, 0])
    return images, labels

def convert_image(image):
    # image = cv2.resize(image, (100, 100),0,0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image

def read_image(filename, image_size):
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_size * 8, image_size * 12), 0, 0, cv2.INTER_LINEAR)
    return image