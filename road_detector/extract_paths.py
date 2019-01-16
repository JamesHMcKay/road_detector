import numpy as np
import os
import glob
import cv2
import sys
import argparse
import math
import matplotlib.pyplot as plt
import yaml

class Link(object):
    def __init__(self, p1, p2):
        self._point_one = p1
        self._point_two = p2

    @property
    def point_one(self):
        return self._point_one

    @point_one.setter
    def point_one(self, value):
        if (len(value) != 2):
            raise ValueError("link point must be a two dimensional array")
        self._point_one = value

    @property
    def point_two(self):
        return self._point_two

    @point_two.setter
    def point_two(self, value):
        if (len(value) != 2):
            raise ValueError("link point must be a two dimensional array")
        self._point_two = value

    def get_length(self):
        diff_x = (self._point_one[0] - self._point_two[0]) ** 2
        diff_y = (self._point_one[1] - self._point_two[1]) ** 2
        return math.pow(diff_x + diff_y, 0.5)

    # id must be the same regardless of link direction
    def get_id(self):
        x1 = int(self._point_one[0])
        x2 = int(self._point_one[1])
        x = str(x1) + "x" + str(x2) if x1 > x2 else str(x2) + "x" + str(x1)
        y1 = int(self._point_two[0])
        y2 = int(self._point_two[1])
        y = str(y1) + "y" + str(y2) if y1 > y2 else str(y2) + "y" + str(y1)
        return y + x


class Vertex(object):
    def __init__(self, origin):
        self._x = origin[0]
        self._y = origin[1]
        self._id = str(origin[0]) + "x" + str(origin[1]) + "y"
        self._adjacent_points = []

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def id(self):
        return self._id

    @property
    def adjancent_points(self):
        return self._adjacent_points

    def add_adjaceny(self, point_id):
        if (point_id not in self._adjacent_points):
            self._adjacent_points.append(point_id)


def prob_road(image, x, y):
    image = image / 255.0
    max_x = x + 1
    max_y = y + 1
    min_x = x - 1
    min_y = y - 1
    if x + 1 >= image.shape[0]:
        max_x = image.shape[0] - 1
    if y + 1 >= image.shape[1]:
        max_y = image.shape[1] - 1
    if x - 1 < 0:
        min_x = 0
    if y - 1 < 0:
        min_y = 0
    prob = 0.11111111 * (
        4.0 * image[x, y] +
        (5.0/8.0) * (
            image[max_x, max_y] +
            image[max_x, min_y] +
            image[max_x, y] +
            image[x, max_y] +
            image[min_x, min_y] +
            image[min_x, max_y] +
            image[min_x, y] +
            image[x, min_y]))
    return (1 - prob)

def random_walk(image, initial_point):
    config = yaml.safe_load(open("config.yaml"))
    num_iterations = config['post_processing']['number_of_iterations']
    mean_step_size = config['post_processing']['initial_step_size']
    step_size = config['post_processing']['step_size']
    tolerance_bad_step = config['post_processing']['tolerance_bad_step']
    tolerance_prob = config['post_processing']['tolerance']
    burn_in_length = config['post_processing']['burn_in_length']

    params = []
    params.append(initial_point)
    current_prob = prob_road(image, params[0][0], params[0][1])

    x_limit = image.shape[0]
    y_limit = image.shape[1]

    xplot = []
    yplot = []

    count = 0
    for n in range(num_iterations):
        step_size_x = int(math.ceil(np.random.normal(0, mean_step_size)))
        step_size_y = int(math.ceil(np.random.normal(0, mean_step_size)))
        current_params = params[len(params) - 1]
        x = current_params[0]
        y = current_params[1]
        x_new = x + step_size_x
        y_new = y + step_size_y
        if (n > burn_in_length):
            mean_step_size = step_size

        isInRange = (
            x_new < x_limit - 1
            and y_new < y_limit - 1
            and y_new > 0
            and x_new > 0)
        take_bad_step = np.random.uniform(0, 1) > 1.0 - tolerance_bad_step
        if (isInRange):
            prob = prob_road(image, x_new, y_new)
            if ((prob > current_prob - tolerance_prob or take_bad_step)):
                params.append([x_new, y_new])
                count += 1
                if (
                    n > burn_in_length
                    and not take_bad_step
                    and current_prob > 0.4):
                    xplot.append(-1.0 * x_new)
                    yplot.append(y_new)
                current_prob = prob
    return xplot, yplot

def extract_paths(input_image):
    image = input_image
    original_size = image.shape

    config = yaml.safe_load(open("config.yaml"))
    horizontalDivisor = config['image_processing']['horizontal_divisions']
    verticalDivisor = config['image_processing']['vertical_divisions']
    image_size = config['image_processing']['training_image_size']
    number_of_steps = config['image_classification']['number_of_translations']

    image = cv2.resize(
        image,
        (
            number_of_steps * horizontalDivisor,
            number_of_steps * verticalDivisor),
        0,
        0,
        cv2.INTER_NEAREST)

    xplot = []
    yplot = []

    xplot_sub, yplot_sub = random_walk(image, [20,20])
    xplot = xplot + xplot_sub
    yplot = yplot + yplot_sub
    xplot_sub, yplot_sub = random_walk(
        image,
        [image.shape[0] - 1, image.shape[1] - 1])
    xplot = xplot + xplot_sub
    yplot = yplot + yplot_sub

    xplot_sub, yplot_sub = random_walk(
        image,
        [number_of_steps, number_of_steps])
    xplot = xplot + xplot_sub
    yplot = yplot + yplot_sub

    number_of_points = len(yplot)

    links = []
    total_length = 0
    for i in range(0, number_of_points - 2, 2):
        new_link = Link([xplot[i], yplot[i]], [xplot[i + 1], yplot[i + 1]])
        total_length += new_link.get_length()
        links.append(new_link)

    number_of_links = len(links)
    mean_link_length = total_length / number_of_links

    links.sort(key=lambda x: x.get_length())

    cutoff_length = links[int(round(0.7 * number_of_links))].get_length()
    lower_cutoff_length = 2

    short_links = []
    global vertices
    vertices = {}
    ids = []
    for link in links:
        link_id = link.get_id()
        if (link.get_length() < cutoff_length
                and link_id not in ids
                and link.get_length() > lower_cutoff_length):
            short_links.append(link)
            ids.append(link_id)
            vertex_one = Vertex(link.point_one)
            vertex_two = Vertex(link.point_two)
            vertex_one.add_adjaceny(vertex_two.id)
            vertex_two.add_adjaceny(vertex_one.id)
            if (vertex_one.id not in vertices):
                vertices[vertex_one.id] = vertex_one
            else:
                vertices[vertex_one.id].add_adjaceny(vertex_two.id)
            if (vertex_two.id not in vertices):
                vertices[vertex_two.id] = vertex_two
                vertices[vertex_two.id].add_adjaceny(vertex_one.id)

    def get_point_span(p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        points = []
        if (x1 == x2):
            upper = y2 if y2 > y1 else y1
            lower = y1 if y2 > y1 else y2
            for t in range(int(lower), int(upper)):
                points.append([x1, t])
        else:
            grad = (y2 - y1) / (x2 - x1)
            upper = x2 if x2 > x1 else x1
            c = y1 if x2 > x1 else y2
            lower = x1 if x2 > x1 else x2
            for t in range(int(lower), int(upper)):
                y = grad * (t - lower) + c
                points.append([t, y])
        return points

    final_result = np.ones(image.shape)
    final_result = final_result * 255
    for link in short_links:
        p1 = link.point_one
        p2 = link.point_two
        points = get_point_span(p1, p2)
        for point in points:
            x = final_result.shape[0] + int(point[0])
            y = final_result.shape[1] - int(point[1])
            final_result[-int(point[0]), int(point[1])] = 0
    return final_result