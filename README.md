# kogakuin_picamera

English:

This program uses a TensorFlow classifier to perform object detection on Picamera feed. Lidar data input from a GR-PEACH microcontroller is given in the form of tuples designating distinct objects, which this program marks as either high priority or low priority (action or ignore).

This code was written at Kogakuin University in Tokyo, Japan for an undergraduate research project in 2019 working on an indoor guide robot, under the guidance of Professor Koyo Katsura, and in partnership with Renesas Electronics Corp.

The overall structure of the code was copied from Evan Juras's sample code implementing Tensorflow object recognition on the RPi:

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/

日本語：

本プログラムはTensorFlowの分類機で物体検出・認識を Picameraの映像で実行する。GR-PEACHに処理されたLidarデータもインプットとなる。そのデータでは、障害物は個別に順序組（tuple）に分かれており、 それらの順序組が高優先か底優先（すなわち待機するか回避するか）と本プログラムに指定される。

本プログラムは、2019年、工学院大学の機械理工学科生の「自動走行案内ロボット」の研究のために書きました。この研究は、ルネサスエレクトロニクス株式会社の連携と、桂晃洋教授の指導の下で行われました。

本プログラムはEvan Jurasさんの、TensorFlow物体認識を用いたサンプルコードをベースにしております。下記のリンクをご覧ください：

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/
