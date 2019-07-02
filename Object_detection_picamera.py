######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description:
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# suppress warning messages about memory allocation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 400   # Use smaller resolution for
IM_HEIGHT = 304  # slightly faster framerate

# Horizontal angular size of the camera in degrees
#IM_ANGLE = 165 # * np.pi / 180 # for the fisheye camera lens
IM_ANGLE = 62.2 # * np.pi / 180 # for the stock picamera

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

# minimum confidence for object detection
MIN_CONF = 0.40

# minimum distance threshold for robot taking allocation
MIN_DIST = 2

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

# dict containing all categories, keyed by the id field of each category
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
# font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize Picamera, grab reference to raw capture, and perform object detection.
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    t1 = cv2.getTickCount()

    # eventually will replace below with something that actually gets the values

    # # tuple containing leftmost angle, rightmost angle, and minimum radius to
    # # a detected object
    # lidar_input = {(np.pi/4, 7*np.pi/4, 0.47)} # in (radians, radians, meters)
    lidar_input = {(-30, 0, 0.47), (0, 30, 0.47)} # in degrees

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    print('\nNew Frame')

    # iterate through the scores and print data corresponding to scores that meet the threshold
    for idx, s in enumerate(scores[0]):
        if s > MIN_CONF:

            print(str(category_index[int(classes[0][idx])]))
            print('confidence: ' + str(scores[0][idx]))
            print('bound: ' + str(boxes[0][idx]))

            for (lidar_angle_l, lidar_angle_r, dist) in lidar_input:

                print(str((lidar_angle_l, lidar_angle_r, dist)))

                # convert box boundaries into angles, where 0 degrees is at the
                # middle of the image
                box_angle_l = (boxes[0][idx][1] - 0.5) * IM_ANGLE
                box_angle_r = (boxes[0][idx][3] - 0.5) * IM_ANGLE
                print('L:' + str(box_angle_l))
                print('R:' + str(box_angle_r))

                # if the closest point on an obstacle is less than MIN_DIST away
                if dist <= MIN_DIST:
                    print('here1')
                    # if the detected boundary box surrounds the lidar reading (???)
                    # what's another way to do this??
                    if box_angle_l < lidar_angle_l and box_angle_r > lidar_angle_r:
                        # if the object is a human
                        if int(category_index[int(classes[0][idx])].get('id'))==1:
                            print('Person detected. Waiting.')
                            print('人間発見。一時待機します。')
                        else:
                            print('Non-person obstacle detected. Rerouting.')
                            print('人間でない障害物発見。回避します。')

    # # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #    frame,
    #    np.squeeze(boxes),
    #    np.squeeze(classes).astype(np.int32),
    #    np.squeeze(scores),
    #    category_index,
    #    use_normalized_coordinates=True,
    #    line_thickness=8,
    #    min_score_thresh=MIN_CONF)
    #
    # cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    #
    # # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    # find FPS, print into console
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    print("FPS: " + str(frame_rate_calc))

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()

cv2.destroyAllWindows()
