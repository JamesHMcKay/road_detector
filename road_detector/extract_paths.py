import numpy as np
import os
import glob
import cv2
import sys
import argparse
import math
import matplotlib.pyplot as plt
from rdp import rdp


# compute the probability that this patch is a road of normal width
# we will take the road to be 3 pixels across
# so this will compute for an effective circle of radius 3/2
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

    # def get_length(self):
    #     diff_x = (self._point_one[0] - self._point_one[1]) ** 2
    #     diff_y = (self._point_two[0] - self._point_two[1]) ** 2
    #     return math.pow(diff_x + diff_y, 0.5)

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
    # prob = image[x, y]
    return (1 - prob)


def compute_orthogonal_distance(upper, lower, point):
    upper = np.array(upper)
    lower = np.array(lower)
    magnitude_a = pow(
        (upper[0] - lower[0]) ** 2 +
        (upper[1] - lower[1]) ** 2, 0.5)
    magnitude_b = pow(
        (point[0] - lower[0]) ** 2 +
        (point[1] - lower[1]) ** 2, 0.5)
    a = upper - lower
    b = point - lower
    a_dot_b = a[0] * b[0] + a[1] * b[1]
    dist = 0
    if (a_dot_b != 0):
        theta = math.acos(a_dot_b / (magnitude_a * magnitude_b))
        dist = magnitude_a * math.sin(theta)
    return dist


def dg_simplification(connected_group):
    global vertices

    arr = []
    for vertex_id in connected_group:
        vertex = vertices[vertex_id]
        arr.append([vertex.x, vertex.y])
    arr = np.array(arr)
    mask = rdp(arr, epsilon=3, algo="iter", return_mask=True)
    return arr[mask]



def extract_paths(input_image):
    image = input_image
    original_size = image.shape
    image = cv2.resize(image, (5 * 8, 5 * 12), 0, 0, cv2.INTER_NEAREST)
    print(image.shape)
    min_value = np.min(image)
    max_value = np.max(image)
    # image = abs(image - max_value)
    print(np.min(image))
    print(np.max(image))

    image = image[2:image.shape[0] - 3, 2:image.shape[1] - 3]

    cv2.imwrite('cut.jpg', image)

    # perform random walk through parameter space
    num_iterations = 500000
    params = []
    params.append([0, 20])

    current_prob = prob_road(image, params[0][0], params[0][1])
    tolerance = 0.2

    x_limit = image.shape[0]
    y_limit = image.shape[1]
    mcmc = np.zeros(image.shape)
    mean_step_size = 5
    tolerance_bad_step = 0.2
    tolerance_prob = 0.3

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
        # print(x_new, y_new)
        if (n > 1000):
            mean_step_size = 3
            tolerance_bad_step = 0.0
            tolerance_prob = 0.2

        isInRange = (
            x_new < x_limit - 1
            and y_new < y_limit - 1
            and y_new > 0
            and x_new > 0)
        take_bad_step = np.random.uniform(0, 1) > 1.0 - tolerance_bad_step
        allow_tolerance_prob = np.random.uniform(0, 1) > 0.5
        allow = 1 if allow_tolerance_prob else 0
        if (isInRange):
            # prob = (prob_road(image, x_new, y_new) + prob_road(image, x, y)) / 2
            prob = prob_road(image, x_new, y_new)
            if ((prob > current_prob - allow * tolerance_prob or take_bad_step)):
                params.append([x_new, y_new])
                current_prob = prob
                count += 1
                if (n > 1000 and not take_bad_step and current_prob > 0.5):
                    mcmc[x_new, y_new] += 1
                    xplot.append(-1.0 * x_new)
                    yplot.append(y_new)


    # x, y = zip(*params)
    number_of_points = len(yplot)
    print('number of points = ', number_of_points)

    # extract links, and clean up links with long distances between points

    links = []

    total_length = 0

    for i in range(0, number_of_points - 2, 2):
        # new_link = Link([yplot[i], yplot[i+1]], [xplot[i], xplot[i+1]])
        new_link = Link([xplot[i], yplot[i]], [xplot[i + 1], yplot[i + 1]])
        total_length += new_link.get_length()
        links.append(new_link)

    number_of_links = len(links)
    print('number of links = ', number_of_links)
    mean_link_length = total_length / number_of_links
    print('mean link length = ', mean_link_length)

    links.sort(key=lambda x: x.get_length())

    cutoff_length = links[int(round(0.95 * number_of_links))].get_length()
    lower_cutoff_length = 1
    print('cutoff length = ', cutoff_length)

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


    print('number of vertices = ', len(vertices))


    def visit_next_vertex(vertex, connected_list):
        global vertices
        for next_point in vertices[vertex].adjancent_points:
            if next_point not in connected_list:
                connected_list.append(next_point)
                connected_list = visit_next_vertex(
                    vertices[next_point].id,
                    connected_list)
        return connected_list


    vertex_count = 0
    # for vertex in vertices:
    #     if (vertex not in connected_list):
    #         vertex_count += 1

    keys = vertices.keys()
    key = keys[0]
    connected_groups = []
    index = 0
    total_connected_points = 0
    visited_points = []
    while (total_connected_points < len(keys)):
        connected_list = []
        while (key in visited_points):
            key = keys[index]
            index += 1
        connected_list.append(key)
        connected_list = visit_next_vertex(vertices[key].id, connected_list)
        total_connected_points += len(connected_list)
        connected_groups.append(connected_list)
        visited_points = visited_points + connected_list

    connected_groups.sort(key=lambda x: len(x))

    # work on the first connected group
    connected_group = connected_groups[len(connected_groups) - 1]
    print('size of group = ', len(connected_group))


    def distance(point_one, point_two):
        diff_x = (point_one[0] - point_two[0]) ** 2
        diff_y = (point_one[1] - point_two[1]) ** 2
        return math.pow(diff_x + diff_y, 0.5)


    for group in connected_groups:
        print('number in group = ', len(group))
        points_original = dg_simplification(group)
        # choose a start point then find the cloest point,
        # then find the cloest point AFTER that point (d_next > d_prev) and so on
        # if such a point does not exist then go back
        # order points wrt to distance from the first point in the list
        points = list(points_original)
        prev_point = points[0]
        path_one = []
        points.sort(key=lambda x: distance(prev_point, x))
        next_point = points[1]
        current_point = next_point
        done = False
        path_length = 0
        possible_new_points = points
        while not done:
            limit = distance(prev_point, current_point)
            possible_new_points = list(filter(
                lambda x: distance(prev_point, x) > limit,
                possible_new_points))
            possible_new_points.sort(key=lambda x: distance(current_point, x))
            if len(possible_new_points) > 1:
                print(
                    'length ', path_length,
                    ' number of possible points = ',
                    len(possible_new_points))
                next_point = possible_new_points.pop(0)
                path_length = path_length + distance(current_point, next_point)
                path_one.append(next_point)
                prev_point = current_point
                current_point = next_point
            else:
                # start a new path, go back
                done = True
        x = []
        y = []
        print('length of connected path = ', len(path_one))
        for point in path_one:
            x.append(point[0])
            y.append(point[1])
        plt.plot(y, x)

        x = []
        y = []
        for point in points_original:
            x.append(point[0])
            y.append(point[1])
        # plt.scatter(y, x, s=1)

    # plt.axes().set_aspect('equal')
    # plt.savefig('mcmc.pdf')

    print('reduced number of links = ', len(short_links))

    # for link in short_links:
    #     p1 = link.point_one
    #     p2 = link.point_two
    #     plt.plot([p2[1], p1[1]], [p2[0], p1[0]],'black')
    # plt.axes().set_aspect('equal')
    # plt.savefig('mcmc.pdf')

    # find all points that are traversed by a link and color these as roads


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
    print("SHAPE = ", final_result.shape)

    for link in short_links:
        p1 = link.point_one
        p2 = link.point_two
        points = get_point_span(p1, p2)
        for point in points:
            x = final_result.shape[0] + int(point[0])
            y = final_result.shape[1] - int(point[1])
            final_result[-int(point[0]), int(point[1])] = 0


    maximum_value = np.max(mcmc)
    print('mean visits is ', np.mean(mcmc))
    print('most visited pixel has vists = ', maximum_value)
    mcmc = (abs(mcmc - maximum_value) / maximum_value) * 255

    print('number of steps taken = ', len(params))
    cv2.imwrite('smoothed.jpg', final_result)
    return final_result

# input_image_from_file = cv2.imread(sys.argv[1], 0)
# extract_paths(input_image_from_file)