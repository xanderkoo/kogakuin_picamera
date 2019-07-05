######## Object Detection for RPi with Picamera and Lidar Input  #########
#
# Author: Xander Koo ゼンダー・クー
# Date: 2019/7/2
# Description:
# This program uses a TensorFlow classifier to perform object detection on a
# Picamera feed. LIDAR data input from a GR-PEACH microcontroller is given in
# the form of tuples designating distinct objects, which this program marks as
# either high priority or low priority (action or ignore).
# 本プログラムはTensorFlowの分類機で物体検出・認識を Picameraの映像で実行する。GR-PEACHに
# 処理されたLIDARデータもインプットとなる。そのデータでは、障害物は個別に順序組（tuple）に
# 分かれており、 それらの順序組が高優先か低優先（すなわち待機するか回避するか）と本プログラムに
# 指定される。
#
## The overall structure of the code was copied from Evan Juras's sample code
## implementing Tensorflow object recognition on the RPi:
## 本プログラムはEvan Jurasさんの、TensorFlow物体認識を用いたサンプルコードをベースに
## しております。下記のリンクをご覧ください：
## https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/


# Import packages
# パッケージを読み込む
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# for connecting the RPi to the GR-PEACH via I2C
# RPiとGR-PEACHを組み合わせるにはI2Cを使用
import smbus
import time
bus = smbus.SMBus(1)

# for processing the input from the GR-PEACH
import struct

# TODO: update this with the address actually being used by the GR-PEACH
# TODO: GR-PEACHの実際のアドレスを書き込まんと
address = 0x08

# suppress warning messages about memory allocation
# 警告を抑制する
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up camera constants
# カメラの解像度を設定
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 400   # Use smaller resolution for
IM_HEIGHT = 304  # slightly faster framerate

# Horizontal angular size of the camera. Multiply by (np.pi / 180) for radians
# カメラの横視角（度）。ラジアンを使用する場合は、視角と (np.pi / 180) をかけてください
#　IM_ANGLE = 165 # for the fisheye camera lens 広角レンズ用
IM_ANGLE = 62.2 # for the stock picamera 標準カメラ用

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilities
# ユーティリティを読み込む
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

# minimum distance threshold for robot to respond to obstacles
MIN_DIST = 2

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                max_num_classes=NUM_CLASSES, use_display_name=True)

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
try:
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):



        t1 = cv2.getTickCount()

        # Eventually will replace below with something that actually gets the values.
        # Assumes that the Lidar-processing GR-PEACH is going to be able to split the
        # input into discrete detected objects.

        # gets one object detected by the lidar (rpi is the master in this case)
        
        lidar_input = {}
        
        # マイナス値の距離に当たるまでループを続ける
        while true:
            # get 12 bytes at a time from the GR-PEACH via I2C, representing 3 floats (left
            # angle, right angle, distance), 4 bytes each.
            in_sublist = bus.read_i2c_block_data(address, 0, 12)
            
            # process the distance first, as a negative distance will indicate a termination
            # of the sequence
            # 距離の値がマイナスである場合は、
            dist = struct.unpack('<f', struct.pack('4B', in_sublist[8:12]))
            if dist < 0:
                break
            
            # tuple containing leftmost angle, rightmost angle, and minimum radius to
            # a detected object, s.t. 0 deg is the middle of the camera's field of view
            in_tuple = (struct.unpack('<f', struct.pack('4B', in_sublist[0:4]))
                        struct.unpack('<f', struct.pack('4B', in_sublist[4:8]))
                        dist)
            lidar_input.add(in_tuple)
            # uses (degrees, degrees, meters)

            # TODO: end byte(s) would be signalled by a negative distance or something 

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

        # iterate through the scores and print data corresponding to scores that
        # meet the threshold. effectively iterates thru every detected object
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
                        print('obstacle within range')
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

                    # TODO: eventually, we want to just output a boolean as a single
                    # bit, to indicate whether the robot car should wait or reroute

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
except KeyboardInterrupt:
    pass

print('Exiting program')

camera.close()

cv2.destroyAllWindows()
