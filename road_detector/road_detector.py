import os
import cv2
import sys
import train
import classify_image
import extract_paths
import re

print 'Starting road detector'

if (len(sys.argv) == 3):
    if (sys.argv[1] == "train"):
        train.train_model(sys.argv[2])
    elif (sys.argv[1] == "test"):
        image_path = sys.argv[2]
        files = os.listdir(image_path)
        files = list(filter(lambda x: 'jpg' in x and 'aux' not in x, files))
        filenames = list(map((lambda x: re.sub('\.jpg$', '', x)), files))
        for file in filenames:
            print '-----------------'
            print 'classifying image ' + file
            path = image_path + file + '.jpg'
            result = classify_image.classify_image(path)
            cv2.imwrite(file + 'raw.jpg', result)
            print 'processing image ' + file
            processed_image = extract_paths.extract_paths(result)
            cv2.imwrite(file + 'processed.jpg', processed_image)
    elif (sys.argv[1] == "post-process"):
        image_path = sys.argv[2]
        files = os.listdir(image_path)
        files = list(filter(lambda x: 'jpg' in x and 'aux' not in x, files))
        filenames = list(map((lambda x: re.sub('\.jpg$', '', x)), files))
        for file in filenames:
            print '-----------------'
            print 'processing image ' + file
            path = image_path + file + '.jpg'
            input_image_from_file = cv2.imread(path, 0)
            processed_image = extract_paths.extract_paths(
                input_image_from_file)
            cv2.imwrite(file + 'processed.jpg', processed_image)
    elif (sys.argv[1] == 'test-single'):
        result = classify_image.classify_image(sys.argv[2])
        cv2.imwrite('raw.jpg', result)
        print 'processing image'
        processed_image = extract_paths.extract_paths(result)
        cv2.imwrite('processed.jpg', processed_image)
    elif (sys.argv[1] == 'post-process-single'):
        input_image_from_file = cv2.imread(sys.argv[2], 0)
        result = extract_paths.extract_paths(input_image_from_file)
        cv2.imwrite('processed.jpg', result)
else:
    print 'Please provide a command and a path'
