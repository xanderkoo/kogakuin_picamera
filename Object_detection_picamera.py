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


print('Importing packages and setting up constants')

import sys
import warnings

# suppress warnings lol
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Import packages
# パッケージを読み込む
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf

# for connecting the RPi to the GR-PEACH via I2C
# RPiとGR-PEACHを組み合わせるにはI2Cを使用
import smbus
import time
bus = smbus.SMBus(1)

# for processing the input from the GR-PEACH
import struct

# TODO: update this with the address actually being used by the LIDAR GR-PEACH
# TODO: 実際のアドレスを書き込まんと/
address_in = 0x08 # address of LIDAR data GR-PEACH
address_out = 0x60 # address of output GR-PEACH

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

print('Initializing TensorFlow model')

# Name of the directory containing the object detection module we're using
# 使用するモデルのディレクトリー
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
# ワークスペースのディレクトリー
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
# .pbファイルのfrozen detection graphを示すパス。このファイルには物体検出のモデルがある
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
# ラベルマップのファイルパス
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
# モデルが検出できるクラス数
NUM_CLASSES = 90

# minimum confidence for object detection
# 認識の最低確率（これ以下の値は表紙しない）
MIN_CONF = 0.65

# minimum distance threshold for robot to respond to obstacles
# 障害物に反応する最低距離
MIN_DIST = 2

## Load the label map.
## ラベルマップを読み込む
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                max_num_classes=NUM_CLASSES, use_display_name=True)

# dict containing all categories, keyed by the id field of each category
# クラスを全部含んだ連想配列（dict）。キー：クラスのID
category_index = label_map_util.create_category_index(categories)

print('Loading model into memory')

# Load the Tensorflow model into memory.
# モデルをメモリーに読み込む
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# 正直わからん
print('Defining tensors')
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
# フレームレートの計算を起動させる
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize Picamera, grab reference to raw capture, and perform object detection.
# Picamera を起動させ、物体認識を実行
print('Initializing Picamera')
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

# object detection
# 物体検出・認識の部分
try:
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        # この部分は具体的に何を行なっているのかわからないが、推論用の同位列(array)を用意しているのではないかと
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Assumes that the Lidar-processing GR-PEACH is going to be able to split the
        # input into discrete detected objects, and that the RPi is the master of
        # the GR-PEACH in question
        # 下記の部分は、GR-PEACHが個別の障害物を検出できる・RPiがマスターで、GR-PEACHがスレーブだであるという前提で書いた
        lidar_input = set()

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # the lines below are for the demo
        # 下記の部分は発表用・デモ用です
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # For the demo, we suppose that there are three objects detected by the
        # LIDAR, one each on the left third, middle third, and right third.
        # At this point, the distance value (0.47) is not relevant, as long as
        # it is less than 2.0
        # 発表の為に、LIDARが検出した物体が三つ（左側、中央、右側）あるという前提で行きます。
        # 今の時点では、距離の値(0.47)は、2.0より低ければ、何も影響もないから重要ではない
        lidar_input = {(-30, -20, 0.47), (-5, 5, 0.47), (20, 30, 0.47)}
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # the section below is commented out for the July demo
        # 七月の発表・デモのために、下記の部分を全部コメントアウトしました
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # TODO: verify if the below gets ALL data.
        # # 下記のコマンドはデータを全部とるかを確認する
        # print(bus.read_i2c_block_data(address_in, 0))
        #
        # # continue looping until it hits a negative value for the distance
        # # マイナス値の距離に当たるまでループを続ける
        # while True:
        #     # get 12 bytes at a time from the GR-PEACH via I2C, representing 3 floats
        #     # (left angle, right angle, distance), 4 bytes each.
        #     # (note: not sure what the second parameter (long cmd) represents
        #
        #     in_list = bus.read_i2c_block_data(address_in, 0, 12)
        #     print(str(in_list))
        #
        #     # process the distance first, as a negative distance will indicate a termination
        #     # of the sequence
        #     # 距離の値がマイナスである場合は、
        #     dist = struct.unpack('<f', struct.pack('4B', *in_list[8:12]))[0]
        #     if dist < 0:
        #         break
        #
        #     # tuple containing leftmost angle, rightmost angle, and minimum radius to
        #     # a detected object, s.t. 0 deg is the middle of the camera's field of view
        #     # (＜最左の角度＞、＜最右の角度＞、＜LIDARと障害物の間の距離＞)の値を含んだ順序組(tuple)
        #     # 0度は画像の中央である
        #     in_tuple = (struct.unpack('<f', struct.pack('4B', *in_list[0:4]))[0],
        #                 struct.unpack('<f', struct.pack('4B', *in_list[4:8]))[0],
        #                 dist)
        #     lidar_input.add(in_tuple)
        #     # uses (degrees, degrees, meters)
        #     # 単位：(度、度、メートル)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Perform the actual detection by running the model with the image as input
        # 画像をインプットとして、推論（物体検出＋認識）を実行する
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # 物体認識の画像が必要とされる場合は、下の部分の上・下の　"""　を抜いてください

        # """
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
           frame,
           np.squeeze(boxes),
           np.squeeze(classes).astype(np.int32),
           np.squeeze(scores),
           category_index,
           use_normalized_coordinates=True,
           line_thickness=8,
           min_score_thresh=MIN_CONF)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        # """


        print('\nNew Frame')

        # iterates thru every TensorFlow detected object
        # TensorFlowの推論の結果をみて、TFが検出した物を一つ一つ処理する
        for idx, s in enumerate(scores[0]):

            # if the TF-detected object has a high enough confidence, continue
            # to see if it corresponds with a LIDAR-detected object
            # 物体の検出確率が最低限より高かったら、処理を続ける
            if s > MIN_CONF:

                # debug用のprint
                # print(str(category_index[int(classes[0][idx])]))
                # print('confidence: ' + str(scores[0][idx]))
                # print('bound: ' + str(boxes[0][idx]))

                # iterates through every element in the lidar_input set, i.e.
                # every distinct object identified by the LIDAR
                for (lidar_angle_l, lidar_angle_r, dist) in lidar_input:

                    # print(str((lidar_angle_l, lidar_angle_r, dist)))

                    # convert box boundaries into angles, where 0 degrees is at the
                    # middle of the image
                    # note: this assumes the use of a fisheye lens, i.e. that angular
                    # diameter is linearly related to distance in the image
                    box_angle_l = (boxes[0][idx][1] - 0.5) * IM_ANGLE
                    box_angle_r = (boxes[0][idx][3] - 0.5) * IM_ANGLE
                    # print('L:' + str(box_angle_l))
                    # print('R:' + str(box_angle_r))

                    # デモ用です
                    if lidar_angle_l == -30:
                        print('Left 左側')
                    if lidar_angle_l == -5:
                        print('Center 中央')
                    if lidar_angle_l == 20:
                        print('Right 右側')

                    # if the closest point on an obstacle is less than MIN_DIST away
                    if dist <= MIN_DIST:
                        # debug用のprint
                        # print('obstacle within range')

                        # If the bounds of the LIDAR-detected object are entirely
                        # within the TensorFlow bounding box, then we will consider
                        # them to be representing the same object
                        # TODO: Figure out a more reliable way of deciding how
                        # to map LIDAR objects to bounding boxes from Tensorflow.
                        #
                        # LIDARが特定した物体の範囲が全部TensorFlowが特定したバウンディング
                        # ボックスの範囲以内であれば、LIDARが検出したものとTFが検出したものは、
                        # 同じ物体を示していると言えるでしょう
                        # TODO: もっとマシ・有効な手段を考えておこう。LIDARの物体とTFの
                        # 物体はどうすればマッピングできるのか？というのが問題です
                        if box_angle_l < lidar_angle_l and box_angle_r > lidar_angle_r:
                            # if the object is a human, send a signal to the GR-PEACH
                            # to wait, and stop looking at other
                            if int(category_index[int(classes[0][idx])].get('id'))==1:
                                print('Person detected. Waiting.')
                                print('人間発見。一旦待機します。')

                                # TODO: transmit True to GR-PEACH

                                # once there is a human within range, we can stop
                                # looping through all the LIDAR-detected objects
                                break
                            else:
                                print('Non-person obstacle detected. Rerouting.')
                                print('人間でない障害物発見。回避します。')
                        else:
                            print('No obstacle detected.')
                            print('物体が検出されていません。')

                # continue looping if inside loop wasn't broken out of
                else:
                    continue

                # if break was called (i.e. if person was detected in range),
                # breaks out of the loop iterating through the TF-detected objects
                break

        # find FPS, print into console
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        print("FPS: " + str(frame_rate_calc))

        rawCapture.truncate(0)

except KeyboardInterrupt:
    pass

print('Exiting program')

camera.close()

cv2.destroyAllWindows()
