#! /usr/bin/python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from pose_msgs.msg import BodypartDetection, PersonDetection  # For pose_msgs

# TRT_pose related
import cv2
import numpy as np
import math
import os
from ros_trt_pose.estimation_utils import preprocess, load_params, load_model, draw_objects